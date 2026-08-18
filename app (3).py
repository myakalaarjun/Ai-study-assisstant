import os
from typing import TypedDict

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# Configuration
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add your Gemini API key as an environment variable."
    )


# ============================================================
# Gemma LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)


# ============================================================
# LangGraph State
# ============================================================

class StudyState(TypedDict, total=False):
    user_request: str
    task: str
    response: str


# ============================================================
# Manager Agent
# ============================================================

def manager_agent(state: StudyState):
    request = state["user_request"]

    prompt = f"""
You are the Manager Agent of an AI Study Assistant.

Classify the student's request into exactly one category.

Categories:
STUDY_PLAN
TEACH
QUIZ
EVALUATE

Rules:
- STUDY_PLAN: timetable, schedule, revision plan, exam preparation plan
- TEACH: explanation of a topic or concept
- QUIZ: MCQs, questions, tests, quizzes
- EVALUATE: checking, scoring, correcting, or improving an answer

Student request:
{request}

Return ONLY one category name.
"""

    result = llm.invoke(prompt)
    task = result.content.strip().upper()

    if "STUDY_PLAN" in task:
        task = "STUDY_PLAN"
    elif "QUIZ" in task:
        task = "QUIZ"
    elif "EVALUATE" in task:
        task = "EVALUATE"
    else:
        task = "TEACH"

    return {"task": task}


# ============================================================
# Specialist Agents
# ============================================================

def study_planner_agent(state: StudyState):
    prompt = f"""
You are the Study Planner Agent.

Create a personalized and realistic study plan for a college student.

Student request:
{state["user_request"]}

Include:
1. Study goals
2. Day-by-day schedule
3. Topics for each day
4. Study hours
5. Revision time
6. Practice questions
7. Break suggestions
8. Final revision

Use simple language and Markdown.
"""
    result = llm.invoke(prompt)
    return {"response": result.content}


def topic_teacher_agent(state: StudyState):
    prompt = f"""
You are the Topic Teacher Agent.

Explain the student's requested topic in simple language.

Student request:
{state["user_request"]}

Include:
1. Simple definition
2. Easy explanation
3. Step-by-step explanation
4. Example
5. Important points
6. Common mistakes
7. Exam-oriented points

If the student asks for a 10-mark answer, provide a
complete structured exam answer.

Use Markdown.
"""
    result = llm.invoke(prompt)
    return {"response": result.content}


def quiz_agent(state: StudyState):
    prompt = f"""
You are the Quiz Generator Agent for college students.

Create a quiz based on:
{state["user_request"]}

Generate:

## Section A - MCQs
5 MCQs with four options: A, B, C, D.
Do not reveal answers immediately.

## Section B - Short Answer
3 questions.

## Section C - Long Answer
2 questions.

## Answer Key
Give the correct answers at the end.

Use Markdown and make the questions educational.
"""
    result = llm.invoke(prompt)
    return {"response": result.content}


def evaluator_agent(state: StudyState):
    prompt = f"""
You are an Answer Evaluation Agent.

Evaluate the student's answer.

Student request:
{state["user_request"]}

Give:
1. Score out of 10
2. Correct points
3. Missing points
4. Incorrect points
5. Improved answer
6. Suggestions for improvement

Be encouraging but accurate.
Use Markdown.
"""
    result = llm.invoke(prompt)
    return {"response": result.content}


# ============================================================
# LangGraph Routing
# ============================================================

def route_task(state: StudyState):
    task = state.get("task", "TEACH")

    if task == "STUDY_PLAN":
        return "planner"
    if task == "QUIZ":
        return "quiz"
    if task == "EVALUATE":
        return "evaluator"

    return "teacher"


graph_builder = StateGraph(StudyState)

graph_builder.add_node("manager", manager_agent)
graph_builder.add_node("planner", study_planner_agent)
graph_builder.add_node("teacher", topic_teacher_agent)
graph_builder.add_node("quiz", quiz_agent)
graph_builder.add_node("evaluator", evaluator_agent)

graph_builder.add_edge(START, "manager")

graph_builder.add_conditional_edges(
    "manager",
    route_task,
    {
        "planner": "planner",
        "teacher": "teacher",
        "quiz": "quiz",
        "evaluator": "evaluator",
    },
)

graph_builder.add_edge("planner", END)
graph_builder.add_edge("teacher", END)
graph_builder.add_edge("quiz", END)
graph_builder.add_edge("evaluator", END)

study_graph = graph_builder.compile()


# ============================================================
# Main Agent Function
# ============================================================

def run_study_agent(user_message: str):
    if not user_message or not user_message.strip():
        return "Please enter a question."

    try:
        result = study_graph.invoke({
            "user_request": user_message.strip()
        })
        return result.get("response", "No response was generated.")
    except Exception as exc:
        return f"### Error\n\n```text\n{exc}\n```"


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="AI Study Assistant Agent") as app:

    gr.Markdown(
        """
        # 🎓 AI Study Assistant Agent

        **Powered by Gemma + LangChain + LangGraph**

        Your personal AI assistant for learning, planning,
        quizzes, and answer evaluation.
        """
    )

    with gr.Tab("🤖 AI Assistant"):
        user_input = gr.Textbox(
            label="Ask your Study Assistant",
            placeholder=(
                "Example: Explain normalization in DBMS "
                "or create a 5-day study plan for Statistics"
            ),
            lines=5,
        )

        ask_button = gr.Button(
            "🚀 Ask AI Agent",
            variant="primary",
        )

        agent_output = gr.Markdown()

        ask_button.click(
            fn=run_study_agent,
            inputs=user_input,
            outputs=agent_output,
        )

    with gr.Tab("📚 Learn Topic"):
        topic_input = gr.Textbox(
            label="Topic",
            placeholder="Example: Explain hypothesis testing",
            lines=3,
        )

        topic_button = gr.Button(
            "📖 Explain Topic",
            variant="primary",
        )

        topic_output = gr.Markdown()

        def learn_topic(topic):
            return run_study_agent(f"Explain this topic: {topic}")

        topic_button.click(
            fn=learn_topic,
            inputs=topic_input,
            outputs=topic_output,
        )

    with gr.Tab("📅 Study Planner"):
        plan_input = gr.Textbox(
            label="Study Requirements",
            placeholder=(
                "Example: I have a DBMS exam in 5 days "
                "and can study 4 hours per day."
            ),
            lines=5,
        )

        plan_button = gr.Button(
            "📅 Create Study Plan",
            variant="primary",
        )

        plan_output = gr.Markdown()

        def create_plan(details):
            return run_study_agent(f"Create a study plan: {details}")

        plan_button.click(
            fn=create_plan,
            inputs=plan_input,
            outputs=plan_output,
        )

    with gr.Tab("📝 Quiz Generator"):
        quiz_input = gr.Textbox(
            label="Quiz Topic",
            placeholder="Example: Statistics",
            lines=3,
        )

        quiz_button = gr.Button(
            "📝 Generate Quiz",
            variant="primary",
        )

        quiz_output = gr.Markdown()

        def create_quiz(topic):
            return run_study_agent(f"Create a quiz about {topic}")

        quiz_button.click(
            fn=create_quiz,
            inputs=quiz_input,
            outputs=quiz_output,
        )

    with gr.Tab("✅ Answer Evaluator"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="Enter your exam question",
            lines=3,
        )

        answer_input = gr.Textbox(
            label="Your Answer",
            placeholder="Enter your answer",
            lines=8,
        )

        evaluate_button = gr.Button(
            "✅ Evaluate My Answer",
            variant="primary",
        )

        evaluation_output = gr.Markdown()

        def evaluate_answer(question, answer):
            request = f"""
Evaluate my answer.

Question:
{question}

Student Answer:
{answer}
"""
            return run_study_agent(request)

        evaluate_button.click(
            fn=evaluate_answer,
            inputs=[question_input, answer_input],
            outputs=evaluation_output,
        )


# ============================================================
# Start server
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
