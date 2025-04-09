"""
model benchmark script to test SageMaker endpoint with whisper model.
"""
import logging
import json
import soundfile as sf
from datasets import load_dataset
import time
from sagemaker.predictor import Predictor
import sagemaker
import boto3
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer

ENDPOINT_NAMES = [
    # "whisper-large-v3-gpu-1742380315-g4dn",
    # "whisper-large-v3-gpu-1742372287-g5",
    # "whisper-large-v3-gpu-1742288179-g6"
    "pytorch-inference-neuronx-ml-inf2-2025-04-09-09-30-09-099"   
]


def invoke_endpoint(predictor, audio_path):
    """
    Invoke SageMaker endpoint with audio file and return transcription result.
    
    Args:
        predictor: SageMaker predictor object
        audio_path (str): Path to audio file
        
    Returns:
        str: Transcribed text
    """
    response = predictor.predict(data=audio_path)
    result = json.loads(response[0])
    return result

def load_audio_dataset():
    """
    Load dataset from Hugging Face
    """
    dataset = load_dataset('MLCommons/peoples_speech', "microset", split='train')
    return dataset


def get_predictor(endpoint_name):
    """
    Get SageMaker predictor object
    """
    boto_session = boto3.Session(region_name="us-west-2")
    sess = sagemaker.Session(boto_session=boto_session)
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sess,
        serializer=DataSerializer(),
        deserializer=JSONDeserializer()
    )
    return predictor

def benchmark_endpoint(endpoint_name, dataset):
    """
    Benchmark SageMaker endpoint with audio dataset
    """
    predictor = get_predictor(endpoint_name)
    start_time = time.time()
    
    csv_filename = f"{endpoint_name}_results.csv"
    with open(csv_filename, 'w') as csv_file:
        csv_file.write("index,transcription,ground_truth,time_seconds,audio_duration_seconds\n")
    
    for i, _item in enumerate(dataset):
        iter_start_time = time.time()
        audio_data = _item["audio"]["array"]
        audio_path = f"sample_audio.wav"
        sf.write(audio_path, audio_data, _item["audio"]["sampling_rate"])
        _response = invoke_endpoint(predictor, audio_path)
        _time = time.time() - iter_start_time
        logging.info(f"Transcription: {_response}, ground truth: {_item['text']}, time: {_time}")
        # Save results to CSV
        # Prepare CSV columns
        index = i
        transcription = _response.replace('"', '""')
        ground_truth = _item["text"].replace('"', '""')
        elapsed_time = _time
        audio_duration = _item["duration_ms"]/1000

        # Create CSV line with clear column mapping
        csv_line = f'{index},"{transcription}","{ground_truth}",{elapsed_time},{audio_duration}\n'
        with open(csv_filename, 'a') as csv_file:
            csv_file.write(csv_line)
    
    total_time = time.time() - start_time
    logging.info(f"Total time: {total_time}")
    logging.info(f"The result has been saved at {csv_filename}.")


def main():
    dataset = load_audio_dataset()
    for endpoint_name in ENDPOINT_NAMES:
        benchmark_endpoint(endpoint_name, dataset)


if __name__ == "__main__":
    main()
    