# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np



# Usage results_to_csv(clf.predict(X_test))
def results_to_csv(y_test,name):
    y_test = np.array(y_test, dtype=np.float32)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    name_file  = name + '.csv'
    df.to_csv(name_file, index_label='Item')