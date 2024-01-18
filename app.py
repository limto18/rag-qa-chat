from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores.chroma import Chroma
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler

import chainlit as cl

from scrape_data import *

model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Load vector database that was persisted earlier
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=STORAGE_PATH, embedding_function=embedding)


@cl.on_chat_start
async def on_chat_start():

    retriever = vectordb.as_retriever()

    template = """Utilisez les éléments de contexte suivants pour répondre à la question finale.
                Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
                Dites toujours "Avez vous d'autre question?" à la fin de la réponse.

                    {context}

                Question : {question}

                    Réponse utile :"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)

    msg = cl.Message(
        content=f"Vous pouvez poser vos questions sur les articles suivants {ARTICLE_URLS}",
    )
    await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("rag_chain")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = [] 

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_doc = d.page_content +"\nSource: "+ d.metadata['source']
                self.sources.append(source_doc)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                #Display the reference docs with a Text widget
                sources_element = [cl.Text(name=f"source_{idx+1}", content=content) for idx,content in enumerate(self.sources)]
                source_names = [el.name for el in sources_element]
                self.msg.elements += sources_element
                self.msg.content += f"\nSources: {', '.join(source_names)}"
                
    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()
