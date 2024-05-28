The below example will use a SQLite connection with Chinook database. Follow these installation steps to create Chinook.db in the same directory as this notebook:

Save this file as Chinook_Sqlite.sql
Run sqlite3 Chinook.db
Run .read Chinook_Sqlite.sql
Test SELECT * FROM Artist LIMIT 10;




### How to get Started?

1. Clone the GitHub repository

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
3. Get your OpenAI API Key

- Sign up for an [OpenAI account](https://platform.openai.com/) (or the LLM provider of your choice) and obtain your API key.

4. Run the Streamlit App
```bash
streamlit run llama3_local_rag.py
```