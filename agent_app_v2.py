from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
from scrape_data import *

model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Load vector database that was persisted earlier
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=STORAGE_PATH, embedding_function=embedding)

@cl.on_chat_start
async def on_chat_start():

    retriever = vectordb.as_retriever()

    condense_question_template = """
                A partir de la conversation suivante et d'une question de suivi, reformulez la question de suivi pour en faire une question indépendante sans changer le contenu de la question donnée.

                Historique de la conversation : {chat_history}

                Question de suivi : {question}

                Question autonome :"""
    
    combine_docs_template = """Utilisez les éléments de contexte suivants pour répondre à la question finale.
                                Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

                                Context:    {context}

                                Question : {question}

                                    Réponse utile :"""
    
    combine_docs_custom_prompt = PromptTemplate.from_template(combine_docs_template)

    condense_question_custom_prompt = PromptTemplate.from_template(condense_question_template)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )


    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_question_custom_prompt,
        combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt)
    )

    msg = cl.Message(
        content=f"Vous pouvez poser vos questions sur les articles suivants: {ARTICLE_URLS}"
    )
    await msg.send()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
