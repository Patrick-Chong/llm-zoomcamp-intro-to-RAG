
from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)

INSTRUCTIONS = f'''Your task is to answer questions from the course participants based on the provided context. 

Use the context to find relevant informaitona nd provide acurate answers. 

If the answer is not found in the context, respond with "I do not know"
'''

USER_PROMPT_TEMPLATE = '''
Question: 
{question}

Context:
{context}
'''

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model


    def search(self, question, course='llm-zoomcamp'):
        boost_dict={'question':2.0, 'section':0.5}
        filter_dict={'course':course}
        
        return self.index.search(question,
                                boost_dict=boost_dict,
                                filter_dict=filter_dict,
                                num_results=5)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["course"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip() 

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip()

    def llm(self, prompt):
        response = self.llm_client.responses.create(
            model="gpt-5.4-mini",
            input=prompt
        )
        return response.output_text

    def rag(self, query, model="gpt-5.4-mini"):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer