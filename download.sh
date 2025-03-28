#!/bin/bash

# S3からモデルファイルをダウンロード
aws s3 cp s3://sagemaker-us-west-2-392304288222/whisper-neuron-model/model.tar.gz ./tmp/tmp-model.tar.gz

# ダウンロードが成功したか確認
if [ $? -eq 0 ]; then
    echo "モデルファイルのダウンロードが完了しました"
    
    # tarファイルを展開
    tar -xzf model.tar.gz
    
    if [ $? -eq 0 ]; then
        echo "モデルファイルの展開が完了しました"
        # 不要になったtar.gzファイルを削除
        rm model.tar.gz
    else
        echo "モデルファイルの展開に失敗しました"
        exit 1
    fi
else
    echo "モデルファイルのダウンロードに失敗しました"
    exit 1
fi