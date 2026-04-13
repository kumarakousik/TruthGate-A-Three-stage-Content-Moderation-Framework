from AgenticRAGState import AgenticRAGState
from Sequential_execution import node3_retrieval
from Toxic_and_Domain_classifier import node1_toxic_and_domain_classifier
from Genrate_querys_by_LLM import node2_generate_query
from Final_judgment import node4_verdict
from langgraph.graph import StateGraph, START, END

build_graph = StateGraph(AgenticRAGState)

build_graph.add_node("Node_1_Toxicity_&_Domain_Classifier", node1_toxic_and_domain_classifier)

build_graph.add_node("Node_2_Generate_Queries_by_LLM",node2_generate_query)

build_graph.add_node("Node_3_Retrieval_&_Sequential_Agents",node3_retrieval)
build_graph.add_node( "Node_4_LLM_Verdict_Generation_ChromaDB_Storage",node4_verdict)

build_graph.add_edge(START, "Node_1_Toxicity_&_Domain_Classifier")
build_graph.add_edge("Node_1_Toxicity_&_Domain_Classifier", "Node_2_Generate_Queries_by_LLM")
build_graph.add_edge("Node_2_Generate_Queries_by_LLM", "Node_3_Retrieval_&_Sequential_Agents")
build_graph.add_edge("Node_3_Retrieval_&_Sequential_Agents", "Node_4_LLM_Verdict_Generation_ChromaDB_Storage")
build_graph.add_edge("Node_4_LLM_Verdict_Generation_ChromaDB_Storage", END)

graph = build_graph.compile()
