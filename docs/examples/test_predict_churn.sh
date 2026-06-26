#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# test_predict_churn.sh
#
# Testa o endpoint POST /predict com um payload de cliente
# com alto risco de churn (resultado esperado: churn_prediction=1).
#
# Perfil do cliente:
#   - Novo (1 mês)
#   - Contrato mensal (sem fidelidade)
#   - Internet Fiber optic
#   - Pagamento via electronic check
#   - Sem suporte técnico, segurança ou backup
#   - Fatura digital
#   - Churn Score alto (95)
#   - CLTV baixo (2000)
#   - Churn Label_Yes = 1 (indica histórico de churn)
#
# Uso:
#   ./docs/examples/test_predict_churn.sh
#   ./docs/examples/test_predict_churn.sh https://custom-url.com
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

# URL padrão — altere ou passe como argumento
API_URL="${1:-https://churn.caprica.tech}"

echo "╔══════════════════════════════════════════════╗"
echo "║  Teste de Inferência — Cliente Alto Risco   ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Endpoint: ${API_URL}/predict"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Health check ───────────────────────────────────────────────────
echo "1. Health check..."
HEALTH=$(curl -s "${API_URL}/health")
echo "   Resposta: ${HEALTH}"
echo ""

# ── Predição — cliente com alto risco de churn ─────────────────────
echo "2. POST /predict (cliente alto risco de churn)..."
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
  "Zip Code": 90003,
  "Latitude": 33.964,
  "Longitude": -118.272,
  "Tenure Months": 1,
  "Monthly Charges": 105.0,
  "Total Charges": 105.0,
  "Churn Score": 95,
  "CLTV": 2000,
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
  "Churn Label_Yes": 1,
  "Churn Reason_Extra data charges": 0,
  "Churn Reason_Lack of affordable download/upload speed": 0,
  "Churn Reason_Lack of self-service on Website": 0,
  "Churn Reason_Limited range of services": 0,
  "Churn Reason_Long distance charges": 0,
  "Churn Reason_Network reliability": 0,
  "Churn Reason_Poor expertise of online support": 0,
  "Churn Reason_Product dissatisfaction": 0
}')

# Separa body e HTTP status
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "   HTTP Status: ${HTTP_CODE}"
echo "   Resposta:"
echo "   ${BODY}" | python3 -m json.tool 2>/dev/null || echo "   ${BODY}"
echo ""

# ── Validação ──────────────────────────────────────────────────────
if [ "$HTTP_CODE" = "200" ]; then
  PREDICTION=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['churn_prediction'])" 2>/dev/null || echo "?")
  PROBABILITY=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['churn_probability'])" 2>/dev/null || echo "?")

  echo "╔══════════════════════════════════════════════╗"
  if [ "$PREDICTION" = "1.0" ] || [ "$PREDICTION" = "1" ]; then
    echo "║  ✅ CHURN DETECTADO                         ║"
  else
    echo "║  ⚠️  Churn NÃO detectado (inesperado)       ║"
  fi
  echo "║  Probabilidade: ${PROBABILITY}              "
  echo "║  Predição:      ${PREDICTION}               "
  echo "╚══════════════════════════════════════════════╝"
else
  echo "╔══════════════════════════════════════════════╗"
  echo "║  ❌ ERRO — HTTP ${HTTP_CODE}                ║"
  echo "╚══════════════════════════════════════════════╝"
  exit 1
fi
