import os
import boto3
import mlflow
import logging
from datetime import datetime

class AutomatedTrainingPipeline:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.model_bucket = 'nifty500-model-registry'
        self.data_bucket = 'nifty500-data-pipeline'
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def check_data_freshness(self):
        """Check if new data is available"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.data_bucket,
                Key='data/processed/last_updated.txt'
            )
            last_modified = response['LastModified']
            
            # Check if data was updated in the last 24 hours
            current_time = datetime.now(last_modified.tzinfo)
            time_diff = current_time - last_modified
            
            return time_diff.total_seconds() < 86400  # 24 hours
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return False
    
    def run_training_pipeline(self):
        """Execute the complete training pipeline"""
        try:
            self.logger.info("Starting automated training pipeline...")
            
            # 1. Data preprocessing
            self.logger.info("Running data preprocessing...")
            os.system("python src/data_preprocessing.py")
            
            # 2. Feature engineering
            self.logger.info("Running feature engineering...")
            os.system("python src/feature_engineering.py")
            
            # 3. Model training
            self.logger.info("Training models...")
            os.system("python src/model_training.py")
            os.system("python train_regime_model.py")
            
            # 4. Model validation
            self.logger.info("Validating models...")
            validation_result = self.validate_models()
            
            if validation_result['passed']:
                # 5. Deploy models
                self.logger.info("Deploying models...")
                self.deploy_models()
                
                # 6. Update model registry
                self.update_model_registry()
                
                self.logger.info("Training pipeline completed successfully!")
                return True
            else:
                self.logger.error("Model validation failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False
    
    def validate_models(self):
        """Validate trained models"""
        # Add your model validation logic here
        return {'passed': True, 'metrics': {}}
    
    def deploy_models(self):
        """Deploy models to S3"""
        try:
            # Upload models to S3
            for root, dirs, files in os.walk('models/'):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_path = local_path.replace('models/', 'models/')
                    
                    self.s3_client.upload_file(
                        local_path,
                        self.model_bucket,
                        s3_path
                    )
                    
            self.logger.info("Models deployed to S3 successfully")
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            raise
    
    def update_model_registry(self):
        """Update model registry with new model information"""
        model_info = {
            'timestamp': datetime.now().isoformat(),
            'version': os.environ.get('GITHUB_SHA', 'unknown'),
            'model_paths': {
                'cagr_model': 'models/best_cagr_model.pkl',
                'regime_model': 'models/regime_detection/gmm_model.pkl'
            }
        }
        
        # Save model registry info
        import json
        with open('model_registry.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Upload to S3
        self.s3_client.upload_file(
            'model_registry.json',
            self.model_bucket,
            'model_registry.json'
        )

if __name__ == "__main__":
    pipeline = AutomatedTrainingPipeline()
    
    # Check if retraining is needed
    if pipeline.check_data_freshness():
        pipeline.run_training_pipeline()
    else:
        print("No new data available, skipping training")
