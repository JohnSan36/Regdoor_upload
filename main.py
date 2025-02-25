from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime
load_dotenv(find_dotenv())
import os

informações_necessarias = """
# Data 
- Deve ser capturada automaticamente com base no dia em que a interação ocorreu, e não quando foi registrada, e caso o usuário fornecer termos relativos como "ontem" ou "amanhã", o você deve pedir uma data específica para evitar ambiguidade.
# Contatos 
- Se apenas "reguladores" ou um órgão regulador for mencionado sem especificar indivíduos, você deve perguntar os nomes completos e as funções.Se apenas um primeiro ou último nome for fornecido, você deve adotar uma abordagem conversacional para extrair os detalhes completos do contato.
# Função
- Se a função de um contato for mencionada de forma incompleta (por exemplo, "consultor", "gerente"), o Agente de IA deve pedir o título completo (por exemplo, "Consultor Chefe de Políticas" em vez de apenas "Consultor").
- Se a função não estiver clara ou estiver faltando, o Agente de IA deve solicitar ao usuário que especifique a posição da pessoa dentro de sua organização.
- Se o usuário fornecer apenas uma designação geral como "executivo" ou "oficial", o Agente de IA deve esclarecer a função do indivíduo (por exemplo, regulatória, conformidade, política, jurídica).
- Se o contato for de um órgão regulador, confirme se ele está envolvido na formulação de políticas, supervisão ou aplicação da lei.
- Se o contato for de uma empresa, confirme se sua função é em conformidade, política, jurídica ou relações governamentais.
- Se o contato for de uma associação, confirme se ele é um representante da indústria, formulador de políticas ou especialista em defesa.
# Organizações
- Se o usuário mencionar apenas a jurisdição, abreviação ou um nome parcial de um órgão regulador, você deve confirmar o nome completo da organização e garantir a especificidade da jurisdição.
# Jurisdição
- Se estiver faltando, você deve confirmar o país ou região relacionada ao órgão ou entidade reguladora.
# Representantes da Empresa
- Se o usuário não mencionar explicitamente representantes da empresa, você deve solicitar que confirmem se alguém de sua empresa participou.
# Assunto
- Extraído diretamente da entrada do usuário, mas deve ser claro e conciso. Se vago, você deve primeiro dar uma opção e pedir esclarecimentos se não for aprovado pelo Usuário.
# Conteúdo
- Deve resumir os principais pontos da discussão. Se faltarem detalhes, solicite ao usuário que forneça insights ou conclusões específicas.
# Ações de Acompanhamento
- Se os acompanhamentos forem mencionados, você deve extrair prazos, responsáveis e próximas etapas específicas.
- Se o usuário não mencionar ações de acompanhamento, pergunte se há alguma tarefa a ser concluída.
# Sentimento Geral
- Você deve atribuir uma pontuação de sentimento (positivo, neutro, negativo) com base na entrada do usuário.
"""

exemplos = """
<exemplo 1>
# Correção de Entrada Parcial:
- Usuário: "Fiz uma ligação com um regulador do Reino Unido."
- IA: "Você poderia especificar qual órgão regulador e o nome do indivíduo?"
# Nome da Organização Ausente:
- Usuário: "Encontrei-me com John Doe."
- IA: "Você poderia confirmar qual organização John Doe representa?"
# Confirmando Acompanhamentos:
- Usuário: "Discutimos atualizações de políticas."
- IA: "Alguma ação de acompanhamento ou prazo específico foi mencionado?"
# Garantindo Clareza do Sentimento:
- Usuário: "Foi uma discussão interessante."
- IA: "Você descreveria a interação como positiva, neutra ou negativa em termos de resultado?"
</exemplo 1>

<exemplo 2>
1- Data: Você mencionou "ontem". Poderia me informar a data específica em que a conversa ocorreu?
2- Contatos: Você se referiu a um representante do Banco Central do Brasil. Qual é o nome completo e a função desse representante?
3- Jurisdição: Presumo que estamos falando sobre o Brasil, está correto?
4- Representantes da Empresa: Algum representante da sua empresa participou dessa conversa?
5- Ações de Acompanhamento: Você mencionou que precisamos preparar documentação adicional. Há prazos ou responsáveis para essa tarefa?
</exemplo 2>
"""

exemplos_listas = """
# Usando '-' ao invés de '.'.
1-Fulano de ciclano pelinous;
2-Lorem Ipsum é simplesmente uma simulação de texto da indústria tipográfica e de impressos, e vem sendo utilizado desde o século XVI, quando um impressor desconhecido pegou uma bandeja de tipos e os embaralhou para fazer um livro de modelos de tipos;
3-Se popularizou na década de 60, quando a Letraset lançou decalques contendo passagens de Lorem Ipsum;
4-Existem muitas variações disponíveis de passagens de Lorem Ipsum, mas a maioria sofreu algum tipo de alteração, seja por inserção de passagens com humor, ou palavras aleatórias que não parecem nem um pouco convincentes;
"""

exemplos_listas = """
- User: Today, we participated in an extensive compliance strategy session involving regulatory experts, legal teams, and compliance officers from multiple jurisdictions. The focus was on building a cohesive strategy to handle the rapidly evolving global regulatory landscape surrounding digital assets and fintech solutions. The session kicked off with a discussion on the recent developments from global regulatory bodies, including the Financial Action Task Force, FATF, the European Securities and Markets Authority, ESMA, and the U.S. Securities and Exchange Commission, SEC. The primary concern was ensuring compliance with anti-money laundering, TML, requirements, particularly with the implementation of the Travel Rule and enhanced KYC procedures across different regions. Among the key contributors were Sarah Bennett from the UK Financial Conduct 
Authority, who discussed the challenges in cross-border compliance, Michael Tanoka from Japan's Financial Services Agency, who emphasized the need for fintech innovation alongside regulatory oversight, and David Rodriguez from the USSEC, who focused on recent enforcement actions against non-compliant digital asset firms. One of the key challenges discussed was the inconsistency in regulatory expectations across jurisdictions. While some regulators are pushing for more stringent rules, others are still in the early stages of defining their approach to digital asset oversight. We debated the feasibility of creating a unified compliance framework that could adapt to these varying regulatory environments without imposing excessive operational burdens on fintech companies and digital asset service providers. We also explored the role of artificial intelligence and machine learning in compliance automation. AI-driven monitoring solutions were highlighted as a promising tool for improving fraud detection, automating reporting, and reducing compliance costs. However, concerns were raised regarding the ethical use of AI in compliance, particularly around data privacy and decision-making transparency. Another major topic was tax 
compliance for digital assets. With OECD's push for a global tax framework and initiatives like the Cryptoasset Reporting Framework, CARF, exchanges, and financial institutions 
must prepare for increased scrutiny in tax reporting. The discussion revolved around strategies for ensuring tax compliance without disrupting business operations. To conclude, 
we outlined next steps, which included forming specialized task forces to address specific compliance challenges, improving cross-border regulatory cooperation, and investing in compliance technology solutions. This session was crucial in setting the groundwork for a more standardized and efficient compliance approach.
- AI: Vou precisar de algumas informações adicionais para estruturar os dados corretamente. Você poderia me informar se essa reunião ocorreu hoje, ou em uma data diferente?
- User: Ocorreu hoje.
- AI:  Você mencionou reguladores. Poderia fornecer os nomes completos de Sarah Bennett, Michael Tanoka e David Rodriguez, bem como suas funções?
- User: Estes nomes mesmo. Eles atuan no cargo xxxx e yyyy.
- AI:  Pode confirmar o nome completo das organizações que eles representam?
- User: Claro. xxxxx, yyyyy.
- AI: As informações foram organizadas com sucesso. Se precisar de mais assistência ou tiver outras perguntas, não hesite em me avisar. Aqui esta uma lista com todas elas organizadas:
Data: 24/02/2025
Contatos: Sarah Bennett, Michael Tanoka e David Rodriguez pertencendo aos cargos xxxx, yyyy, zzzzz respectivamente.
E assim por diante, forneça uma lista com todas as informações.
"""


api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)


def obter_hora_e_data_atual():
    """Retorna a hora atual e a data de hoje."""
    agora = datetime.now()
    return agora.strftime("%Y-%m-%d - T%H:%M:%S")
data_atual = obter_hora_e_data_atual()


class ExtraiInformacoes(BaseModel):
    """Extrair informações de complience"""
    data: str = Field(description="data em que o o evento ocorreu (se a pessoa disser algo como ontem ou algo do tipo, perguntar a data especifica)")
    contatos: str = Field(
        description="nome das pessoas contidas no texto", 
        examples=[
            ("Me chamo Rafael e falei com o Junior sobre assunto XYZ.", "Rafael, Junior"), 
            ("Me chamo Alfredo e falei com o Severino sobre assunto a posse dos Eua.", "Alfredo, Severino")
        ])
    meio: str = Field(description="meio de contato dos contatos mencionados. Deve perguntar se foi google meet, presencial, ou qual meio foi utilizado.")
    cargo: str = Field(description="cargo dos contatos mencionados")
    organizacoes: str = Field(description="organizaçao dos contatos mencionados")
    jurisdicoes: str = Field(
        description="jurisdições mencionadas",
        examples=[
            ("Jane Doe is a Senior Regulatory Advisor at the Financial Conduct Authority (FCA) in the UK. I don't have her email ou phone number at the moment. ", "UK")
        ])
    representantes: str = Field(description="representantes dos contatos mencionados")
    assunto: str = Field(description="assunto do texto, deve ser 'politica', 'economia' ou 'justica'.")
    resumo: str = Field(description="resumo do texto, deve ser uma breve descrição do evento, com no máximo 100 caracteres.")
    acoes_acompanhamento: str = Field(description="acoes de acompanhamento do texto.")
    sentimento: str = Field(description="sentimento expresso pelo individuo, deve ser 'positivo', 'negativo' ou 'neutro'.")


@tool(args_schema=ExtraiInformacoes)
def extrutura_informacao(
        data: str, 
        contatos: str, 
        meio: str,
        cargo: str, 
        organizacoes: str, 
        jurisdicoes: str, 
        representantes: str, 
        assunto: str, 
        resumo: str, 
        acoes_acompanhamento: str, 
        sentimento: str):
    
    """Extrutura as informações do texto"""
    return data, contatos, meio, cargo, organizacoes, jurisdicoes, representantes, assunto, resumo, acoes_acompanhamento, sentimento


toolls = [extrutura_informacao]
toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]


prompt = ChatPromptTemplate.from_messages([
    ("system", f"Você é um assistente juridico que extrai informações do texto fornecido apenas quando todas as {informações_necessarias} estiverem presentes, e caso alguma delas não esteja, pergunte ao usuario antes de acionar a tool 'extrutura_informacao'. Pergunte uma coisa de cada vez até que todas as informações estejam presentes e você possa acionar o tool, fornecendo uma lista com todas informações contidas ao final. Para referencia a data atual é {data_atual}. Não utilize formatação markdown. Caso precise, sigo os exemplos em {exemplos}. Não use asteriscos '*' em suas mensagens. Proibido usar asteriscos '*' em suas mensagens. Proibido usar formatação markdown. Quando for listar algo, use '-' ao invés de '.' como nos exemplos {exemplos_listas}. REGRA: Para listar itens use o exemplo de {exemplos_listas}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

pass_through = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
)
agent_chain = pass_through | prompt | chat.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()


def run_agent(input):
    passos_intermediarios = []
    while True:
        resposta = agent_chain.invoke({
            "input": input,
            "agent_scratchpad": format_to_openai_function_messages(passos_intermediarios)
        })
        if isinstance(resposta, AgentFinish):
            return resposta
        observacao = toolls[resposta.tool].run(resposta.tool_input)
        passos_intermediarios.append((resposta, observacao))


agent_executor = AgentExecutor(
    agent=agent_chain,
    memory=memory,
    tools=toolls,
    verbose=True,
    return_intermediate_steps=True
)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def receive_message():
    
    try:
        body = request.data  
        body_str = body.decode("utf-8")
        print("Mensagem recebida:", body_str)  
        resposta = agent_executor.invoke({"input": body_str})
        resposta_final = resposta["output"]

        return jsonify({"Status": resposta_final})

    except Exception as e:
        return jsonify({"Erro": f"Falha ao processar JSON: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)