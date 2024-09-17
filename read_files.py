import pandas as pd
import numpy as np

def process_file(file_path) -> pd.DataFrame:
    save_data = False
    df = pd.DataFrame()
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if 'Average Reynolds #' in line:
                reynolds  = float(lines[i+1])
            
            if 'Number of angles of attack:' in line:
                number_of_aoa = int(lines[i+1])
            
            if 'alpha' in line:
                data = np.array([[float(a.split()[0]), float(a.split()[1]), float(a.split()[2])] for a in lines[i+1: i+1+number_of_aoa]])
                df[f'Re = {reynolds}'] = [data]
            
            i+=1
    
    return df
                
                
                
                

    
    