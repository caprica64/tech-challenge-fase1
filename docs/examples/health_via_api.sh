#! /bin/bash

curl -s -X GET https://churn.caprica.tech/health

## Result should be similar
# {"status":"ok","model_loaded":true}
