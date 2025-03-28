import os
os.environ['NEURON_RT_NUM_CORES']='1'
import types
import torch
import torch_neuronx
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer
from custom_layer import make_forward_neuron
from datasets import load_dataset
import subprocess
import shutil
import boto3

# モデルサイズとバッチサイズの設定
model_size = "large-v3"  # tiny, small, medium, large-v3 から選択
batch_size = 1
suffix = model_size
max_dec_len = 128


def compile_model():
    # モデルのロード
    compile_dir = "./neuron_model"
    model_id = f"openai/whisper-{model_size}"
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id, torchscript=True)
    enc_f, dec_f, proj_out_f = make_forward_neuron(processor)

    # 次元の設定
    if model_size == "tiny":
        dim_enc = 384
    elif model_size == "small":
        dim_enc = 768
    elif model_size == "medium":
        dim_enc = 1024
    elif model_size == "large-v3":
        dim_enc = 1280
    else:
        raise ValueError("サポートされていないモデルサイズです")

    if not hasattr(model.model.encoder, 'forward_'): model.model.encoder.forward_ = model.model.encoder.forward
    if not hasattr(model.model.decoder, 'forward_'): model.model.decoder.forward_ = model.model.decoder.forward
    if not hasattr(model.proj_out, 'forward_'): model.proj_out.forward_ = model.proj_out.forward

    model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
    model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
    model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)

    model.model.decoder.max_length = max_dec_len
    model.proj_out.max_length = max_dec_len
    dim_enc = model.config.num_mel_bins
    dim_dec = model.config.d_model
    # サンプル入力の作成
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # sample #3 is ~9.9seconds and produces 33 output tokens + pad token
    sample = dataset[3]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
    
    # warmup
    print("Warming up the model...")
    model.generate(input_features)
    # エンコーダーのトレース
    print("Tracing the encoder...")
    model_filename = f"{compile_dir}/whisper_{suffix}_{batch_size}_neuron_encoder.pt"
    os.makedirs(compile_dir, exist_ok=True)
    if not os.path.isfile(model_filename):
        inputs = (
            torch.zeros([1, dim_enc, 3000],dtype=torch.float32),
            torch.zeros([1, dim_enc], dtype=torch.int64))
        if hasattr(model.model.encoder, "forward_neuron"): del model.model.encoder.forward_neuron
        neuron_encoder = torch_neuronx.trace(
            model.model.encoder,
            inputs,
            compiler_args="--model-type=transformer --auto-cast=all --auto-cast-type=bf16",
            compiler_workdir="./enc_dir",
            inline_weights_to_neff=False)
        neuron_encoder.save(model_filename)
        model.model.encoder.forward_neuron = neuron_encoder
    else:
        model.model.encoder.forward_neuron = torch.jit.load(model_filename)

    # デコーダーのトレース
    print("Tracing the decoder...")
    model_filename=f"{compile_dir}/whisper_{suffix}_{batch_size}_{max_dec_len}_neuron_decoder.pt"
    if not os.path.isfile(model_filename):
        inputs = (torch.zeros([1, max_dec_len], dtype=torch.int64), torch.zeros([1, 1500, dim_dec], dtype=torch.float32))
        if hasattr(model.model.decoder, 'forward_neuron'): del model.model.decoder.forward_neuron
        neuron_decoder = torch_neuronx.trace(
            model.model.decoder, 
            inputs,
            compiler_args='--model-type=transformer --auto-cast=all --auto-cast-type=bf16',
            compiler_workdir='./dec_dir',      
            inline_weights_to_neff=True)
        neuron_decoder.save(model_filename)
        model.model.decoder.forward_neuron = neuron_decoder
    else:
        model.model.decoder.forward_neuron = torch.jit.load(model_filename)

    print("Tracing the projection layer...")
    model_filename=f"{compile_dir}/whisper_{suffix}_{batch_size}_{max_dec_len}_neuron_proj.pt"
    if not os.path.isfile(model_filename):
        inputs = torch.zeros([1, max_dec_len, dim_dec], dtype=torch.float32)
        if hasattr(model.proj_out, 'forward_neuron'): del model.proj_out.forward_neuron
        neuron_decoder = torch_neuronx.trace(
            model.proj_out, 
            inputs,
            compiler_args='--model-type=transformer --auto-cast=all --auto-cast-type=bf16',
            compiler_workdir='./proj_out_dir',      
            inline_weights_to_neff=True)
        neuron_decoder.save(model_filename)
        model.proj_out.forward_neuron = neuron_decoder
    else:
        model.proj_out.forward_neuron = torch.jit.load(model_filename)
    return compile_dir


def compress_model(compile_dir, fast_compression=True, ultrafast_compression=False):
    """
    モデルディレクトリを圧縮してtarファイルを作成します。
    
    Args:
        compile_dir: コンパイルされたモデルファイルが格納されているディレクトリ
        fast_compression: 高速圧縮モードを使用するか
        ultrafast_compression: 超高速モード（圧縮なし）を使用するか
        
    Returns:
        str: 圧縮されたモデルファイルのパス
    """
    
    output_file = "model.tar.gz"
    
    # neuron_model/codeディレクトリを作成
    code_dir = os.path.join(compile_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # カレントディレクトリの code/ からファイルをコピー
    current_code_dir = os.path.join(os.getcwd(), "code")
    
    # inference.py と requirements.txt をコピー
    shutil.copy(os.path.join(current_code_dir, "inference.py"), os.path.join(code_dir, "inference.py"))
    shutil.copy(os.path.join(current_code_dir, "requirements.txt"), os.path.join(code_dir, "requirements.txt"))
    
    print(f"カレントディレクトリの code/ から inference.py と requirements.txt をコピーしました")
    
    # CPU数を取得して最適なスレッド数を決定
    num_threads = 4
    

    try:
        # pigzが利用可能か確認
        result = subprocess.run(["which", "pigz"], capture_output=True, text=True)
        if result.returncode == 0:
            # 超高速モード: 圧縮なし（-0）、最大スレッド数
            print(f"pigzを使用した並列処理を実行します（超高速モード: 無圧縮、{num_threads}スレッド）")
            tar_cmd = f"tar -cf - -C {compile_dir} . | pigz -0 -p {num_threads} > {output_file}"
            subprocess.run(tar_cmd, shell=True, check=True)
        else:
            # pigzがない場合は、tarのみで圧縮なしで実行
            print("pigzが見つかりませんでした。tar形式のみで圧縮なしで実行します")
            # *.tar.gz という名前だが実際は圧縮なしの*.tarになる
            subprocess.run(["tar", "-cf", output_file, "-C", compile_dir, "."], check=True)
    except Exception as e:
        print(f"超高速モードが失敗しました: {e}")
        print("標準の圧縮方法に戻ります")
        subprocess.run(["tar", "-czf", output_file, "-C", compile_dir, "."], check=True)
    
    
    print(f"モデルを {output_file} に圧縮しました。構造は以下の通りです:")
    print("- code/")
    print("  - inference.py")
    print("  - requirements.txt")
    print("- [モデルファイル(.pt)]")
    
    return output_file

# SageMakerにデプロイするためのコード
def deploy_to_sagemaker(sagemaker_session, role, compressed_file):
    # モデルアーティファクトをS3にアップロード
    print("Uploading model artifact to S3...")
    model_data = sagemaker_session.upload_data(
        path=compressed_file,
        key_prefix="whisper-neuron-model"
    )
    
    
    model = PyTorchModel(
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.5.1-neuronx-py310-sdk2.21.0-ubuntu22.04",
        model_data=model_data,
        entry_point="inference.py",
        model_server_workers=1,
        role=role,
        env={
            "chunk_length_s":"30",
            "MMS_MAX_REQUEST_SIZE": "2000000000",
            "MMS_MAX_RESPONSE_SIZE": "2000000000",
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "900",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1"
        },
        sagemaker_session=sagemaker_session
    )
    model._is_compiled_model = True

    print("Deploying model to SageMaker...")
    # モデルのデプロイ
    predictor = model.deploy(
        instance_type="ml.inf2.xlarge",
        initial_instance_count=1
    )
    
    # プレディクターの設定
    predictor.serializer = DataSerializer()
    predictor.deserializer = JSONDeserializer()
    
    return predictor

# 推論の実行例
def run_inference(predictor, audio_path):
    response = predictor.predict(data=audio_path)
    return response

# メイン実行部分
if __name__ == "__main__":
    # SageMakerにデプロイ
    role = "arn:aws:iam::392304288222:role/service-role/AmazonSageMaker-ExecutionRole-20250130T094469"
    boto_session = boto3.Session(region_name="us-west-2")
    sess = sagemaker.Session(boto_session=boto_session)
    compile_dir = compile_model()
    compressed_file = compress_model(compile_dir)
    predictor = deploy_to_sagemaker(sess, role, compressed_file)
    
    # 推論の実行
    result = run_inference(predictor, "sample_audio.wav")
    print(result)
