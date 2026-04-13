import json

def obter_ultimo_calculo(st_session_state):
    if "ultimo_calculo" in st_session_state:
        try:
            return json.dumps(st_session_state["ultimo_calculo"], ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"erro_interno": f"Falha na leitura dos dados: {str(e)}"})
    else:
        return json.dumps({"aviso": "O usuário ainda não realizou nenhum cálculo na interface gráfica."})