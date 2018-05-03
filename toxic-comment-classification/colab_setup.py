def setup(drive):
    file_import = drive.CreateFile({'id':'1AfKIkqPd3J9RUUuPNUtuj6E6E6Vh9QH9'})
    file_import.GetContentFile('fast_text_embeddings.npy')
    file_import = drive.CreateFile({'id':'1svRXMwRlvbfkubHBRhKKTM6ArKLwDIA1'})
    file_import.GetContentFile('custom_fast_text_embeddings.npy')

    file_import = drive.CreateFile({'id':'1IBd-lqhmNqKz-G3n5naOpflq4aLAOo9z'})
    file_import.GetContentFile('sample_submission.csv')

    file_import = drive.CreateFile({'id':'14BQIrJoMr15-7e1Ijn4JIRs-RP2oSBJh'})
    file_import.GetContentFile('y_train_full.npy')
    file_import = drive.CreateFile({'id':'1TmZd86YdR3_UqGLD-C2dJlca4gSHuE7m'})
    file_import.GetContentFile('y_train.npy')
    file_import = drive.CreateFile({'id':'1JhZvB2sOfcZhNcMFG6JvuJSZawHjiyAV'})
    file_import.GetContentFile('y_val.npy')
    file_import = drive.CreateFile({'id':'13pc10T15c2_fDnE2HqtAlR9FN2A5PDBD'})
    file_import.GetContentFile('y_test.npy')

    file_import = drive.CreateFile({'id':'1pNFVlUX1DP_RAJVOuoC6Ep3fpZ0CV-hL'})
    file_import.GetContentFile('X_train_full.npy')
    file_import = drive.CreateFile({'id':'12d0Qjxrf3xQRyU3umap47Sg3neYPaa0f'})
    file_import.GetContentFile('X_train.npy')
    file_import = drive.CreateFile({'id':'1q0_wlpIl629IeUfH2VqiQ7UD3zAOOAs-'})
    file_import.GetContentFile('X_val.npy')
    file_import = drive.CreateFile({'id':'1a3weNKFr7XMVGOTlaen33ropvFyj4jN6'})
    file_import.GetContentFile('X_test.npy')
    file_import = drive.CreateFile({'id':'1sYvkToH21T8U8dIyO0TBhPeaKr_OkopS'})
    file_import.GetContentFile('X_submission.npy')

    file_import = drive.CreateFile({'id':'1FOfcAROe481NiYkkezqlGJfadYvnPZqF'})
    file_import.GetContentFile('plot_history.py')
    file_import = drive.CreateFile({'id':'1YsDs9rt7I7bSc7UTNFZfoQZcsG66ND_r'})
    file_import.GetContentFile('roc_auc_callback.py')
    file_import = drive.CreateFile({'id':'1QgfZKOZhs49UfptGW2J7iLCelzqsQ1pL'})
    file_import.GetContentFile('train_model.py')

    return