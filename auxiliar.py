from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import nltk
from azure.storage.blob import BlobServiceClient
import yaml
from yaml.loader import SafeLoader
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import pandas as pd
from io import BytesIO


def clustering(df):
    """

    :param df:
    :return:
    """

    try:
        # 3. Converter textos em vetores de características usando TF-IDF
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X = vectorizer.fit_transform(df['Info_Completa'])

        # 4. Aplicar K-means para clusterização
        true_k = 3  # Suponha que queremos 3 clusters
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

        # Adicionar os clusters aos textos originais para visualização
        df['Cluster'] = model.labels_

        # 6. Visualizar usando PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())
        df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
        df_pca['Cluster'] = model.labels_

        # Certifique-se de que a coluna 'Cluster' seja tratada como categoria
        df_pca['Cluster'] = df_pca['Cluster'].astype(str)

        # Usar Plotly para visualização
        fig = px.scatter(
            df_pca,
            x='Principal Component 1',
            y='Principal Component 2',
            color='Cluster',
            title="Clusterização de funcionários",
            labels={"Principal Component 1": "Componente Principal 1", "Principal Component 2": "Componente Principal 2"}
        )

        # Mostrar a legenda dos clusters no gráfico
        fig.update_layout(legend_title_text='Clusters')

        st.plotly_chart(fig)

        st.write("Top termos por cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(true_k):
            st.write(f"Cluster {i}:")
            terms_list = []
            for ind in order_centroids[i, :10]:
                terms_list.append(terms[ind])
            st.write(", ".join(terms_list))

    except Exception as e:
        print("Sem dados suficientes para clusterizar!")

# Baixar o léxico do VADER, se ainda não foi baixado
nltk.download('vader_lexicon')
# Criar instâncias do analisador de sentimentos e do tradutor
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
def traduzir_texto(texto):
    try:
        traducao = translator.translate(texto, src='pt', dest='en')
        return traducao.text
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return texto


def read_yaml_from_blob(container_name, blob_name, storage_account_key):
    """
    Lê um arquivo YAML de um blob no Azure Blob Storage.

    :param container_name: Nome do contêiner no Azure Blob Storage.
    :param blob_name: Nome do blob no Azure Blob Storage.
    :param key_file_path: Caminho para o arquivo de texto contendo a chave de armazenamento. Default é 'storage_key.txt'.
    :return: O conteúdo do arquivo YAML como um dicionário, ou None se houver um erro.
    """
    # Ler a chave de armazenamento do arquivo de texto
    # with open(key_file_path, 'r') as file:
    #     storage_account_key = file.read().strip()
    storage_account_key = storage_account_key
    # Defina o nome da conta de armazenamento
    storage_account_name = "hlbdatalake"

    # Criar a connection string
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"

    # Conectar ao BlobServiceClient usando a connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Criar o ContainerClient
    container_client = blob_service_client.get_container_client(container_name)

    # Criar o BlobClient
    blob_client = container_client.get_blob_client(blob_name)

    # Baixar o arquivo YAML do blob e carregar o conteúdo
    try:
        download_stream = blob_client.download_blob()
        yaml_content = download_stream.readall()
        config = yaml.load(yaml_content, Loader=SafeLoader)
        return config
    except Exception as e:
        print(f"Erro ao ler o arquivo YAML do blob: {e}")
        return None


def read_storage_key(key_file_path):
    # Ler a chave de armazenamento do arquivo de texto
    with open(key_file_path, 'r') as file:
        storage_account_key = file.read().strip()

    return storage_account_key


def list_blobs_in_container(container_name, storage_account_key):
    """
    Lista todos os blobs em um contêiner no Azure Blob Storage.

    :param container_name: Nome do contêiner no Azure Blob Storage.
    :param key_file_path: Caminho para o arquivo de texto contendo a chave de armazenamento. Default é 'storage_key.txt'.
    """


    # Defina o nome da conta de armazenamento
    storage_account_name = "hlbdatalake"

    # Criar a connection string
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    
    # Conectar ao BlobServiceClient usando a connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Criar o ContainerClient
    container_client = blob_service_client.get_container_client(container_name)
    
    # Listar os blobs no contêiner
    try:
        blob_list = container_client.list_blobs()
        print(f"Blobs no contêiner '{container_name}':")
        for blob in blob_list:
            print("\t" + blob.name)
    except Exception as e:
        print(f"Erro ao listar blobs: {e}")


def upload_to_blob(container_name, upload_file_path, directory_name=None, key_file_path='storage_key.txt'):
    """
    Faz upload de um arquivo para o Azure Blob Storage, possivelmente em um diretório especificado.

    :param container_name: Nome do contêiner no Azure Blob Storage.
    :param upload_file_path: Caminho completo do arquivo local que você deseja fazer upload.
    :param directory_name: Nome do "diretório" no Blob Storage onde o arquivo será salvo. Opcional.
    :param key_file_path: Caminho para o arquivo de texto contendo a chave de armazenamento. Default é 'storage_key.txt'.
    """
    # Ler a chave de armazenamento do arquivo de texto
    with open(key_file_path, 'r') as file:
        storage_account_key = file.read().strip()

    # Defina o nome da conta de armazenamento
    storage_account_name = "hlbdatalake"

    # Criar a connection string
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    
    # Conectar ao BlobServiceClient usando a connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Criar o ContainerClient
    container_client = blob_service_client.get_container_client(container_name)
    
    # Definir o nome completo do blob, incluindo o "diretório" se fornecido
    if directory_name:
        blob_name = os.path.join(directory_name, os.path.basename(upload_file_path))
    else:
        blob_name = os.path.basename(upload_file_path)
    
    # Criar o BlobClient
    blob_client = container_client.get_blob_client(blob_name)
    
    # Fazer upload do arquivo
    try:
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data)
        print(f"Arquivo {upload_file_path} enviado com sucesso para {container_name}/{blob_name}.")
    except Exception as e:
        print(f"Erro ao fazer upload do arquivo: {e}")

def read_excel_from_blob(connection_string, container_name, blob_name,cols):
    """
    Lê um arquivo Excel de um contêiner blob do Azure e retorna um DataFrame do pandas.

    :param connection_string: String de conexão ao Azure Blob Storage.
    :param container_name: Nome do contêiner do blob.
    :param blob_name: Nome do arquivo blob.
    :return: DataFrame do pandas com os dados do arquivo Excel.
    """
    # Conectar ao serviço de blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Baixar o blob
    blob_client = container_client.get_blob_client(blob_name)
    stream_downloader = blob_client.download_blob()
    blob_data = stream_downloader.readall()

    # Ler o conteúdo do blob como um DataFrame do pandas
    df = pd.read_excel(BytesIO(blob_data), engine='openpyxl', names=cols)

    return df



