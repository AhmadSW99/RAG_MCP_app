import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain.tools import tool
from dotenv import load_dotenv
import os
from typing import Literal

# Load environment variables
load_dotenv()

# Set your OpenAI API key (Note: Consider using environment variables for security)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the retriever tool
@tool
def retriever_tool(query: str) -> str:
    """Search the story about Daniel's character development, persistence, and hope themes"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./first_Aid_embeddings"
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant story information found."
        
        results = ["Found relevant story excerpts:\n"]
        for i, doc in enumerate(docs, 1):
            results.append(f"Story Excerpt {i}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error accessing story database: {str(e)}"

class JIRAAgentGraph:
    def __init__(self):
        self.client = None
        self.tools = []
        self.llm = None
        self.graph = None
    
    async def initialize_mcp_client(self):
        """Initialize the MCP client and get JIRA tools"""
        print("Initializing MCP client...")
        
        try:
            self.client = MultiServerMCPClient({
                "mcp-atlassian": {
                    "transport": "stdio",
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e", "JIRA_URL",
                        "-e", "JIRA_USERNAME", 
                        "-e", "JIRA_API_TOKEN",
                        "ghcr.io/sooperset/mcp-atlassian:latest"
                    ],
                    "env": {
                        "JIRA_URL": "url",
                        "JIRA_USERNAME": "email",
                        "JIRA_API_TOKEN": "token"
                    }
                },
                "mcp-bitbucket": {
                    "transport": "stdio",
                    "command": "docker",
                    "args": [
                        "run", "-i", "--rm",
                        "-e", "BITBUCKET_USERNAME",
                        "-e", "BITBUCKET_APP_PASSWORD",
                        "mcp-bitbucket"
                    ],
                    "env": {
                        "BITBUCKET_USERNAME": BITBUCKET_USERNAME,
                        "BITBUCKET_APP_PASSWORD": BITBUCKET_APP_PASSWORD
                    }
                }
            })
            
            # Get tools from MCP server with error handling
            print("Loading MCP tools...")
            try:
                mcp_tools = await self.client.get_tools()
                print(f"Successfully loaded {len(mcp_tools)} MCP tools")
            except Exception as tool_error:
                print(f"Warning: Failed to load MCP tools: {tool_error}")
                print("Continuing with retriever tool only...")
                mcp_tools = []
            
            # Combine MCP tools with retriever tool
            self.tools = mcp_tools + [retriever_tool]
            print(f"Total tools available: {len(self.tools)}")
            
            # Initialize LLM with all tools bound
            self.llm = init_chat_model("gpt-4o-mini", model_provider="openai").bind_tools(self.tools)
            
            return self.tools
            
        except Exception as e:
            print(f"Error initializing MCP client: {e}")
            # Fallback to just the retriever tool
            print("Falling back to retriever tool only...")
            self.tools = [retriever_tool]
            self.llm = init_chat_model("gpt-4o-mini", model_provider="openai").bind_tools(self.tools)
            return self.tools
    
    def create_agent_graph(self):
        """Create the LangGraph agent with nodes"""
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Define the agent node
        def agent_node(state: MessagesState) -> MessagesState:
            """Agent node that processes messages and decides on actions"""
            messages = state["messages"]
            
            # Add system message for context
            system_message = SystemMessage(
                content="""You are an intelligent assistant with access to JIRA, Bitbucket, and story analysis capabilities.
You can help users with:

Bitbucket Operations:
- Searching repositories
- Listing repositories in projects
- Getting repository information
- Repository management tasks
- When working with Bitbucket:
  * Always include workspace="ahmadalawwad" unless specified)
  * For project searches, use format: project.key = "PROJECT_NAME"
  * Default to pagelen=10 and page=1 if not provided

JIRA Operations:
- Listing issues/tickets
- Creating new issues  
- Updating issue status
- Searching for specific issues
- Getting project information

Bitbucket Operations:
- Searching repositories
- Listing repositories in projects
- Getting repository information
- Repository management tasks
- When working with Bitbucket:
  * Always include workspace="ahmadalawwad" unless specified)
  * For project searches, use format: project.key = "PROJECT_NAME"
  * Default to pagelen=10 and page=1 if not provided

Story Analysis:
- Search Daniel's story for character development, persistence, and hope themes
- Answer questions based on story excerpts
- Provide insights from the story database

Cross-Platform Analysis:
- Correlate JIRA tickets with repository activities
- Analyze development workflows across both platforms
- Provide insights combining project management and code repository data

Always be helpful and provide clear, structured responses. When using tools, ensure all required parameters are included."""
            )
            
            # Combine system message with user messages
            all_messages = [system_message] + messages
            
            # Get response from LLM
            response = self.llm.invoke(all_messages)
            
            return {"messages": [response]}
        
        # Define routing logic
        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            """Determine whether to continue with tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the last message has tool calls, go to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Otherwise, end
            return "__end__"
        
        # Build the graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.add_edge(START, "agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        
        # After tools, go back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
        
        return self.graph
    
    async def run_agent(self, user_input: str):
        """Run the agent with user input"""
        if not self.graph:
            raise ValueError("Graph not initialized. Call create_agent_graph() first.")
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return result["messages"][-1].content
    
    async def close(self):
        """Clean up resources"""
        if self.client:
            # Check if the client has a close method before calling it
            if hasattr(self.client, 'close'):
                try:
                    await self.client.close()
                except Exception as e:
                    print(f"Warning: Error closing MCP client: {e}")
            elif hasattr(self.client, 'cleanup'):
                try:
                    await self.client.cleanup()
                except Exception as e:
                    print(f"Warning: Error cleaning up MCP client: {e}")
            else:
                print("MCP client doesn't have a close method - skipping cleanup")

# Interactive mode function
async def interactive_mode():
    """Run the agent in interactive mode"""
    agent_graph = JIRAAgentGraph()
    
    try:
        print("Initializing RAG with MCP...")
        await agent_graph.initialize_mcp_client()
        agent_graph.create_agent_graph()
        
        print("Agent initialized successfully!")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("-" * 50)
        
        while True:
            user_input = input("ASK: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                print("Agent: ", end="")
                response = await agent_graph.run_agent(user_input)
                print(response)
                print()
            except Exception as e:
                print(f"Error processing request: {e}\n")
    
    except Exception as e:
        print(f"Initialization error: {e}")
    
    finally:
        try:
            await agent_graph.close()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

async def main():
    """Main function to demonstrate the agent"""
    await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())