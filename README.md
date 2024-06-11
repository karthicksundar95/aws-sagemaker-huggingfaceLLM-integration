# aws-sagemaker-huggingfaceLLM-integration
Repository to show how to connect and integrate and deploy huggingface LLM models as endpoint on AWS sagemaker 

# AWS Sagemaker & Hugging Face Integration for Question Answering Service

This project demonstrates how to use AWS Sagemaker to integrate with Hugging Face to create an endpoint for any Large Language Model (LLM) and build a question-answering service. The project includes an IPython notebook created on AWS Sagemaker Jupyter Studio.

## Overview

- **AWS Sagemaker**: A service that helps users build, train, and deploy machine learning models.
- **User Access**: Provides an API endpoint for users to invoke and get results.
- **Pricing**: Pay-as-you-go model; charges are based on the number of requests made.

## Steps to Setup and Deploy

### AWS Sagemaker Setup

1. **Setup for Single User**
   - Click on "Setup for single user".
   - Wait until the setup is complete.
2. **Select User Role**
   - Click on the domain name to select the user role.
   - In the next stage, select the user role.
3. **Create Jupyter Studio Space**
   - Click on the +create icon -> Jupyter Studio.
   - On the Jupyter Studio screen, click on JupyterLab.
   - Click on +Create JupyterLab Space.
   - Provide a name, type of instance, and image configuration.
   - Click on "Create Space". A new space will be created.
   - Click on "Open JupyterLab".

### Sagemaker Setup Code

1. **Install Sagemaker**
   - Once the notebook is open, install Sagemaker:
     ```python
     !pip install sagemaker -U
     ```
2. **Create Sagemaker Session, Session Bucket, and IAM Role**
   - Initialize the Sagemaker session and set up the session bucket and IAM role:
     ```python
     import sagemaker
     from sagemaker import get_execution_role

     # Initialize a SageMaker session
     sagemaker_session = sagemaker.Session()

     # Create a session bucket
     session_bucket = sagemaker_session.default_bucket()

     # Get the execution role
     iam_role = get_execution_role()

     # Reinitialize the SageMaker session with the created session bucket
     sagemaker_session = sagemaker.Session(default_bucket=session_bucket)
     ```

### Hugging Face Setup Code

1. **Set Hub Model Configuration**
   - From `hf.co/models`, choose the model configuration:
     - Model ID: `distilbert-base-uncased-distilled-squad`
     - Model Type: `question answering`
2. **Create Hugging Face Class Object**
   - Pass the Hub model configuration, IAM role with permission to create endpoint, and versions of transformer, PyTorch, and Python:
     ```python
     from sagemaker.huggingface import HuggingFaceModel

     # Define the model
     huggingface_model = HuggingFaceModel(
         model_id="distilbert-base-uncased-distilled-squad",
         role=iam_role,
         transformers_version="4.6",
         pytorch_version="1.7.1",
         py_version="py36",
     )

     # Deploy the model
     predictor = huggingface_model.deploy(
         initial_instance_count=1,
         instance_type="ml.m5.xlarge"
     )
     ```

3. **Deploy Hugging Face Model and Create a Predictor Object**
   - Pass the context (from which LLM should learn) along with the question prompt to the predictor object to get the response from the model:
     ```python
     # Define the context and question
     context = "Your context here."
     question = "Your question here."

     # Get the response
     response = predictor.predict({
         'inputs': {
             'question': question,
             'context': context
         }
     })

     print(response)
     ```

## Conclusion

This README provides a step-by-step guide to setting up and deploying a question-answering service using AWS Sagemaker and Hugging Face. For detailed instructions and code, please refer to the IPython notebook included in the repository.

---

For further information, please refer to the official documentation of [AWS Sagemaker](https://aws.amazon.com/sagemaker/) and [Hugging Face](https://huggingface.co/).

