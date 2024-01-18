# Création D'un Agent Conversationnel Sécialisé

Ce repo présente la construction d'un agent conversationnel capable de répondre aux questions sur une base de documents spécifique en utilisant la technique de la génération augmenté de récupération (Retrieval Augmented Generation -[RAG](https://www.elastic.co/fr/what-is/retrieval-augmented-generation) en anglais).


## Etapes de Réalisations

1. **Préparation des données:** Nous avons tout d'abord scraper, traiter et stocker quelques articles du site [Ecofin](https://www.agenceecofin.com/a-la-une/recherche-article?filterTitle=&submit.x=0&submit.y=0&filterTousLesFils=Tous&filterCategories=Sous-rubrique&filterDateFrom=&filterDateTo=&option=com_dmk2articlesfilter&view=articles&filterFrench=French&Itemid=269&userSearch=1&layout=#dmk2articlesfilter_results) 📚 afin de construire notre base de documents.

2.  **Creation d'une chaine de traitement:** Ensuite nous avons créer une chaine de traitement du RAG!   

3. **Interface web:** Finalement le tout a été incorporer dans une interface web conviviale en utilisant `Chainlit` pour interagir avec l'agent.💬

## Fonctionnement global de l'agent

- Recuperation des articles sous forme de documents
- Découper ces documents en morceaux et créer des embeddings.
- Stocker ces embeddings sur Chroma
- Répondre aux questions des utilisateurs et montrer les sources utilisées pour répondre.

## Comment faire fonctionner le code

1. Installer les dépendances de Python :
```shell
pip install -r requirements.txt
```

2. Renomer le fichier  `.env.example` en `.env` et inserer votre clé d'api openai:
```.env
OPENAI_API_KEY=VOTRE_OPENAI_API_KEY
```
3. Executer le script pour scraper et stoker les données:
```shell
python scrape_data.py
```
4. Lancez la démo et commencer à interagir avec l'agent. Cela ouvrira une fenêtre de navigateur avec l'interface Chainlit.

```shell
chainlit run app.py 
```

## Réference
Ce code est principalement basé sur le [cookbook/chroma-qa-chat](https://github.com/Chainlit/cookbook/tree/main/chroma-qa-chat) de `chainlit`.