from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, LiteLLMModel

class EventPlanningRetrieverTool(Tool):
    name = "event_planning_retriever"
    description = "Uses semantic search to retrieve party ideas for a speakeasy-styled party for my anniversary."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning and speakeasy themes.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

party_ideas = [
    {"text": "A 1920s speakeasy-themed party with a hidden entrance, password entry, and vintage Prohibition-era glamour throughout.", "source": "Party Ideas 1"},
    {"text": "A live jazz trio or swing band performing Prohibition-era standards to drive the evening's atmosphere.", "source": "Entertainment Ideas"},
    {"text": "A burlesque or cabaret performance as a mid-night show moment to shift the energy of the room.", "source": "Entertainment Ideas"},
    {"text": "A tarot card reader or fortune teller stationed in a corner booth for guests to drift toward.", "source": "Entertainment Ideas"},
    {"text": "A casino corner with blackjack or poker tables using prop chips for a Prohibition-era gaming feel.", "source": "Entertainment Ideas"},
    {"text": "A code word game where guests learn a phrase at the door and use it at the bar for a bonus drink or perk.", "source": "Entertainment Ideas"},
    {"text": "An unmarked entrance concept using a bookshelf or vending machine that opens into the party space.", "source": "Decoration Ideas"},
    {"text": "Low lighting in brass and amber tones paired with exposed brick or dark wood paneling.", "source": "Decoration Ideas"},
    {"text": "Vintage furniture such as leather chesterfields, velvet booths, and antique mirrors throughout the venue.", "source": "Decoration Ideas"},
    {"text": "Subtle smoke or haze effects used to soften the lighting and enhance the speakeasy mood.", "source": "Decoration Ideas"},
    {"text": "Newspaper-print and Prohibition-era signage, such as 'Prohibition Enforced Here,' used as decor accents.", "source": "Decoration Ideas"},
    {"text": "Classic Prohibition-era cocktails like the Sidecar, Bee's Knees, Last Word, and Old Fashioned.", "source": "Catering Ideas"},
    {"text": "A mixologist performing tableside shaking and theatrics rather than running a standard open bar.", "source": "Catering Ideas"},
    {"text": "Small plates such as deviled eggs, oysters, charcuterie, and mini sliders instead of a full sit-down dinner.", "source": "Catering Ideas"},
    {"text": "A house cocktail branded as 'bathtub gin,' served in mason jars or teacups for authenticity.", "source": "Catering Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Creating the retrieval tool
event_planning_retriever = EventPlanningRetrieverTool(docs_processed)

model = LiteLLMModel(model_id="ollama/qwen2.5-coder:14b",
                    api_base="http://localhost:11434",
                    api_key="ollama"
                    )

# Initialize the agent
agent = CodeAgent(tools=[event_planning_retriever], model=model)

response = agent.run(
    "Find ideas for a speakeasy-styled party, including entertainment, catering, and decoration options."
)

print(response) 



