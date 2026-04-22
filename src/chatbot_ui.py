#!/usr/bin/env python3
"""
PoBot Chatbot UI - Simple web interface using Gradio
Run: python chatbot_ui.py
Then open: http://localhost:7860
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from datetime import datetime

# Import RAG pipeline
from rag_pipeline import ask, vectorstore


def chat_with_pobot(message, history):
    """
    Process user message and return PoBot response.
    
    Args:
        message: User's question
        history: Chat history (list of [user, bot] pairs)
    
    Returns:
        Bot's response string
    """
    if not message or not message.strip():
        return "Please ask a question about Hong Kong labor law."
    
    try:
        # Run RAG query
        result = ask(message)
        answer = result['result']
        
        # Format sources
        sources = []
        for doc in result['source_documents']:
            source_name = doc.metadata.get('source', 'Unknown')
            category = doc.metadata.get('category', 'N/A')
            sources.append(f"📄 {source_name} ({category})")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)
        
        # Format response with sources
        formatted_response = f"{answer}\n\n---\n**Sources:**\n" + "\n".join(unique_sources)
        
        return formatted_response
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease try rephrasing your question."


def sample_question_handler(message):
    """Handle sample question button clicks."""
    return message


# Sample questions for quick testing
SAMPLE_QUESTIONS = [
    "What are the rights of Foreign Domestic Helpers?",
    "What is the minimum wage in Hong Kong?",
    "Do domestic workers get rest days?",
    "What happens if an agency overcharges?",
    "Are FDHs required to live with employers?",
    "What laws must employment agencies follow?",
]


def create_ui():
    """Create and launch the Gradio interface."""
    
    # Welcome message
    WELCOME = (
        "Hello! I'm PoBot, your Hong Kong labor law assistant. 🇭🇰\n\n"
        "I can help you with questions about:\n"
        "• Foreign Domestic Helpers (FDH) rights\n"
        "• Employment agencies and commission rules\n"
        "• Minimum wage and salary protection\n"
        "• Rest days, holidays, and leave\n"
        "• Employment contracts and termination\n\n"
        "Ask me anything below!"
    )
    
    with gr.Blocks(
        title="PoBot - HK Labor Law Assistant",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🤖 PoBot - Hong Kong Labor Law Assistant
            
            Ask questions about employment rights, wages, rest days, and more.
            Answers are based on official Hong Kong Labour Department documents.
            """
        )
        
        # Chat interface - using simple Chatbot with tuples
        chatbot = gr.Chatbot(
            label="PoBot",
            height=450,
            show_copy_button=True,
        )
        
        # Input row
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask about HK labor law (e.g., 'What are the rights of FDHs?')...",
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
        
        # Sample questions
        gr.Markdown("\n**💡 Try these questions:**")
        with gr.Row():
            for i, question in enumerate(SAMPLE_QUESTIONS[:3]):
                btn = gr.Button(question, size="sm", variant="secondary")
                btn.click(
                    fn=sample_question_handler,
                    inputs=[gr.Textbox(value=question, visible=False)],
                    outputs=[msg_input]
                )
        
        with gr.Row():
            for i, question in enumerate(SAMPLE_QUESTIONS[3:]):
                btn = gr.Button(question, size="sm", variant="secondary")
                btn.click(
                    fn=sample_question_handler,
                    inputs=[gr.Textbox(value=question, visible=False)],
                    outputs=[msg_input]
                )
        
        # Sidebar info
        with gr.Accordion("ℹ️ About PoBot", open=False):
            gr.Markdown(
                f"""
                **Data Sources:**
                - Employment Ordinance
                - Minimum Wage Ordinance  
                - FDH Employment Guide
                - Employment Agency Codes
                
                **Session Info:**
                - Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                - Documents indexed: 10
                
                **Note:** This is an AI assistant. For legal advice, consult the Labour Department.
                """
            )
        
        # Event handlers
        def respond(message, chat_history):
            """Generate response and update chat history."""
            if not message:
                return "", chat_history
            
            bot_message = chat_with_pobot(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        # Submit button
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        # Enter key to submit
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        # Clear button
        clear_btn.click(
            fn=lambda: ("", []),
            inputs=[],
            outputs=[msg_input, chatbot]
        )
        
        # Initialize chatbot with welcome
        demo.load(
            fn=lambda: ("", [(WELCOME, None)]),
            inputs=[],
            outputs=[msg_input, chatbot]
        )
    
    return demo


if __name__ == "__main__":
    # Check prerequisites
    if not os.path.exists("../vectorstore/faiss_index"):
        print("❌ ERROR: Vectorstore not found. Run embedding_setup.py first.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🤖 PoBot Chatbot UI")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:7860 in your browser\n")
    
    # Launch
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
