

# Construir o sistema de dicionarios e listas com as mensagens
import streamlit as st
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
import pandas as  ps
ps.read_csv("base_dados.csv")

#inf="base_dados.csv"
dt=ps.read_csv("base_dados.csv")
print(dt)

vectorizador=TfidfVectorizer()
x=vectorizador.fit_transform(dt["perguntas"])
y=dt["respostas"]

mirai=MultinomialNB()
mirai.fit(x,y)

nova_pergunta=["E qeum é então suposto ser essa tal de Sanura de que tanto se fala?"]
x_nova=vectorizador.transform(nova_pergunta)
resposta=mirai.predict(x_nova)

print("Pergunta:",nova_pergunta[0])
print("Resposta:",resposta[0])
mirai 

st.write("Oi, eu sou a mirai, pode perguntar oque quiser")
#st.title("Fale comigo sobre a Sanura")

if not "mensagens enviadas" in st.session_state:
    st.session_state["mensagens enviadas"]=[]

for mensagem in st.session_state["mensagens enviadas"]:
    role=mensagem["role"]
    content=mensagem["content"]
    st.chat_message(role).write(content)



mensagem_usuario = st.chat_input("pergunte algo simples")

# Só executa o restante se o usuário mandar uma mensagem
if mensagem_usuario:
    st.chat_message("user").write(mensagem_usuario)
    mensagem = {"role": "user", "content": mensagem_usuario}
    st.session_state["mensagens enviadas"].append(mensagem)

    # Geração da resposta
    x_nova = vectorizador.transform([mensagem_usuario])
    resposta_ia = mirai.predict(x_nova)[0]

    import time
    time.sleep(2.5)

    st.chat_message("assistant").write(resposta_ia)
    mensagem_ia = {"role": "assistant", "content": resposta_ia}
    st.session_state["mensagens enviadas"].append(mensagem_ia)

print(dt.columns)
