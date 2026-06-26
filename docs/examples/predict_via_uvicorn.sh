#! /bin/bash

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
  "Zip Code": 90003,
  "Latitude": 33.964,
  "Longitude": -118.272,
  "Tenure Months": 2,
  "Monthly Charges": 95.0,
  "Total Charges": 190.0,
  "Churn Score": 86,
  "CLTV": 3239,
  "Gender_Male": 1,
  "Senior Citizen_Yes": 0,
  "Partner_Yes": 0,
  "Dependents_Yes": 0,
  "Phone Service_Yes": 1,
  "Multiple Lines_No phone service": 0,
  "Multiple Lines_Yes": 0,
  "Internet Service_Fiber optic": 1,
  "Internet Service_No": 0,
  "Online Security_No internet service": 0,
  "Online Security_Yes": 0,
  "Online Backup_No internet service": 0,
  "Online Backup_Yes": 0,
  "Device Protection_No internet service": 0,
  "Device Protection_Yes": 0,
  "Tech Support_No internet service": 0,
  "Tech Support_Yes": 0,
  "Streaming TV_No internet service": 0,
  "Streaming TV_Yes": 0,
  "Streaming Movies_No internet service": 0,
  "Streaming Movies_Yes": 0,
  "Contract_One year": 0,
  "Contract_Two year": 0,
  "Paperless Billing_Yes": 1,
  "Payment Method_Credit card (automatic)": 0,
  "Payment Method_Electronic check": 1,
  "Churn Label_Yes": 0,
  "Churn Reason_Extra data charges": 0,
  "Churn Reason_Lack of affordable download/upload speed": 0,
  "Churn Reason_Lack of self-service on Website": 0,
  "Churn Reason_Limited range of services": 0,
  "Churn Reason_Long distance charges": 0,
  "Churn Reason_Network reliability": 0,
  "Churn Reason_Poor expertise of online support": 0,
  "Churn Reason_Product dissatisfaction": 0
}' | python3 -m json.tool

## Result should be similar
# {
#    "churn_probability": 0.4325,
#    "churn_prediction": 0.0
# }
