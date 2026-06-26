#! /bin/bash
curl -s http://localhost:8000/health | python3 -m json.tool


## Result should be similar
# {
#     "status": "ok",
#     "model_loaded": true
# }
