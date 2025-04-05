import pickle
import os
import glob

def check_model(model_file):
    print(f"Checking model file: {model_file}")
    try:
        file_size = os.path.getsize(model_file)
        print(f"File size: {file_size} bytes")
        
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            
        print(f"Model type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"Model keys: {list(model.keys())}")
            
            for key, value in model.items():
                print(f"  {key}: {type(value)}")
                
                if hasattr(value, "feature_importances_") and hasattr(value.feature_importances_, "__getitem__"):
                    print(f"  {key} feature importances sample: {value.feature_importances_[:3]}...")
                    
                if hasattr(value, "n_features_in_"):
                    print(f"  {key} number of features: {value.n_features_in_}")
        else:
            print("Model is not a dictionary")
    except Exception as e:
        print(f"Error processing model file: {e}")
    
    print()

# Get some model files from our recent training
model_files = glob.glob("to/models/*/20230602.pkl")[:3]
for model_file in model_files:
    check_model(model_file)

# Try to load and test a specific model
specific_model_file = os.path.join("to", "models", "TGN共享单车", "20230605.pkl")
if os.path.exists(specific_model_file):
    check_model(specific_model_file)
else:
    print(f"Specific model file not found: {specific_model_file}") 