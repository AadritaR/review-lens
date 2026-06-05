from huggingface_hub import upload_folder

upload_folder(
    folder_path='models/distilbert_sentiment',
    repo_id='Aadritaray/review-lens-sentiment',
    repo_type='model'
)
print('Upload complete')