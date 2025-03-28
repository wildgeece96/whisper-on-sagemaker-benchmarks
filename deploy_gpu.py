import os
import subprocess
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq, WhisperTokenizer
import time
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer


boto_session = boto3.Session(region_name="us-west-2")
sess = sagemaker.Session(boto_session=boto_session)
role = sagemaker.get_execution_role(sagemaker_session=sess)
sess_bucket = sess.default_bucket()


if __name__ == "__main__":
    print(f'sagemaker role arn: {role}')
    print(f'sagemaker bucket: {sess_bucket}')
    print(f'sagemaker session region: {sess.boto_region_name}')
    sagemaker_role = "arn:aws:iam::392304288222:role/service-role/AmazonSageMaker-ExecutionRole-20250130T094469"

    save_dir = "gpu_model"
    os.makedirs(save_dir, exist_ok=True)
    model_name = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    os.makedirs("tmp", exist_ok=True)
    subprocess.run(["cp", "-r", "code_gpu", f"{save_dir}/code"])
    tar_cmd = f"tar -cf - -C {save_dir} . | pigz -0 -p 4 > tmp/model.tar.gz"
    subprocess.run(tar_cmd, shell=True, check=True)

    model_uri = sess.upload_data(
        "tmp/model.tar.gz", bucket=sess_bucket, key_prefix="whisper-large-v3-gpu")
    print(f"Model URI: {model_uri}")

    id = int(time.time())
    model_name = f"whisper-large-v3-gpu-{id}"

    # !Please change the image URI for the region that you are using:e.g. us-east-1
    image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04"

    whisper_hf_model = HuggingFaceModel(
        model_data=model_uri,
        role=sagemaker_role,
        image_uri=image,
        entry_point="inference.py",
        name=model_name,
        sagemaker_session=sess,
        env = {
            "chunk_length_s":"30",
            "MMS_MAX_REQUEST_SIZE": "2000000000",
            "MMS_MAX_RESPONSE_SIZE": "2000000000",
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "900",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1"
        }
    )
    audio_serializer = DataSerializer(content_type="audio/x-audio")
    deserializer = JSONDeserializer()

    endpoint_name_g4dn = f"whisper-large-v3-gpu-{id}-g4dn"
    predictor_g4dn = whisper_hf_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        serializer=audio_serializer,
        deserializer=deserializer,
        endpoint_name=endpoint_name_g4dn
    )
    print(f"Endpoint deployed: {predictor_g4dn.endpoint_name}")

    endpoint_name_g5 = f"whisper-large-v3-gpu-{id}-g5"
    predictor_g5 = whisper_hf_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        serializer=audio_serializer,
        deserializer=deserializer,
        endpoint_name=endpoint_name_g5
    )
    print(f"Endpoint deployed: {predictor_g5.endpoint_name}")

    endpoint_name_g6 = f"whisper-large-v3-gpu-{id}-g6"
    predictor_g6 = whisper_hf_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g6.xlarge",
        serializer=audio_serializer,
        deserializer=deserializer,
        endpoint_name=endpoint_name_g6
    )
    print(f"Endpoint deployed: {predictor_g6.endpoint_name}")

        