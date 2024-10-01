import requests
import boto3
import os
import zipfile


def download_zip(url, local_filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Download do arquivo ZIP {local_filename} concluído.")
    else:
        print(f"Erro ao baixar o arquivo. Erro: {response.status_code}")


def unzip(zip_filename,
          extract_to='.'):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Arquivo {zip_filename} descompactado para {extract_to}")


def upload_to_s3(local_filename,
                 bucket_name,
                 s3_filename):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_filename, bucket_name, s3_filename)
        print(f"Upload do arquivo para o bucket {bucket_name} concluído.")
    except Exception as e:
        print(f"Erro ao fazer upload para o S3: {e}")


enem_zip_url = 'https://download.inep.gov.br/microdados/'\
               'microdados_enem_2023.zip'
local_zip_filename = 'microdados_enem_2023.zip'
extracted_folder = 'enem_data'
csv_file_path = os.path.join(extracted_folder,
                             'DADOS',
                             'microdados_enem_2023.csv')
bucket_name = 'tc3grupo46'
file_key = 'raw/microdados_enem_2023.csv'

download_zip(enem_zip_url, local_zip_filename)

unzip(local_zip_filename, extracted_folder)

upload_to_s3(csv_file_path, bucket_name, file_key)
