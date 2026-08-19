import os
from typing import TypedDict

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# 2. GEMMA MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)


# ============================================================
# 3. CONVERT GEMMA RESPONSE TO TEXT
# ============================================================

def content_to_text(content):

    if content is None:
        return ""

    # Normal string
    if isinstance(content, str):
        return content

    # Gemma may return a list
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if item.get("text"):
                    parts.append(str(item["text"]))

                elif item.get("content"):
                    parts.append(
                        content_to_text(item["content"])
                    )

            else:
                parts.append(str(item))

        return "\n".join(parts)

    # Dictionary response
    if isinstance(content, dict):

        if content.get("text"):
            return str(content["text"])

        if content.get("content"):
            return content_to_text(
                content["content"]
            )

    return str(content)


# ============================================================
# 4. CALL GEMMA SAFELY
# ============================================================

def invoke_llm(prompt):

    result = llm.invoke(prompt)

    return content_to_text(result.content)


# ============================================================
# 5. LANGGRAPH STATE
# ============================================================

class StudyState(TypedDict, total=False):

    user_request: str
    task: str
    response: str


# ============================================================
# 6. MANAGER AGENT
# ============================================================

def manager_agent(state: StudyState):

    request = state["user_request"]

    prompt = f"""
You are the Manager Agent of an AI Study Assistant.

Your job is to identify what the student wants.

Choose exactly ONE category:

STUDY_PLAN
TEACH
QUIZ
EVALUATE

Rules:

STUDY_PLAN:
Student wants a timetable, schedule,
revision plan or exam preparation plan.

TEACH:
Student wants an explanation of a topic.

QUIZ:
Student wants MCQs, questions or a quiz.

EVALUATE:
Student wants their answer checked,
corrected or scored.

Student request:

{request}

Return ONLY the category name.
"""

    task = invoke_llm(prompt)

    task = task.strip().upper()

    if "STUDY_PLAN" in task:

        task = "STUDY_PLAN"

    elif "QUIZ" in task:

        task = "QUIZ"

    elif "EVALUATE" in task:

        task = "EVALUATE"

    else:

        task = "TEACH"

    return {
        "task": task
    }


# ============================================================
# 7. STUDY PLANNER AGENT
# ============================================================

def study_planner_agent(state: StudyState):

    prompt = f"""
You are the Study Planner Agent.

Create a personalized study plan for a college student.

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

Make the plan realistic.

Use simple language and Markdown.
"""

    response = invoke_llm(prompt)

    return {
        "response": response
    }


# ============================================================
# 8. TOPIC TEACHER AGENT
# ============================================================

def topic_teacher_agent(state: StudyState):

    prompt = f"""
You are the Topic Teacher Agent.

Explain the student's topic in very simple language.

Student request:

{state["user_request"]}

Include:

1. Simple definition
2. Easy explanation
3. Step-by-step explanation
4. Real-world example
5. Important points
6. Common mistakes
7. Exam-oriented points

If the student asks for a 10-mark answer,
give a complete structured exam answer.

Use Markdown.
"""

    response = invoke_llm(prompt)

    return {
        "response": response
    }


# ============================================================
# 9. QUIZ AGENT
# ============================================================

def quiz_agent(state: StudyState):

    prompt = f"""
You are the Quiz Generator Agent.

Create a college-level quiz.

Topic/request:

{state["user_request"]}

Generate:

## Section A - MCQs

5 multiple-choice questions.

Each question must have:

A
B
C
D

## Section B - Short Answer

3 questions.

## Section C - Long Answer

2 questions.

## Answer Key

Give the correct answers at the end.

Use Markdown.
"""

    response = invoke_llm(prompt)

    return {
        "response": response
    }


# ============================================================
# 10. ANSWER EVALUATOR AGENT
# ============================================================

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

    response = invoke_llm(prompt)

    return {
        "response": response
    }


# ============================================================
# 11. ROUTER
# ============================================================

def route_task(state: StudyState):

    task = state.get(
        "task",
        "TEACH"
    )

    if task == "STUDY_PLAN":

        return "planner"

    if task == "QUIZ":

        return "quiz"

    if task == "EVALUATE":

        return "evaluator"

    return "teacher"


# ============================================================
# 12. CREATE LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    StudyState
)


# Add agents

graph_builder.add_node(
    "manager",
    manager_agent
)

graph_builder.add_node(
    "planner",
    study_planner_agent
)

graph_builder.add_node(
    "teacher",
    topic_teacher_agent
)

graph_builder.add_node(
    "quiz",
    quiz_agent
)

graph_builder.add_node(
    "evaluator",
    evaluator_agent
)


# START → MANAGER

graph_builder.add_edge(
    START,
    "manager"
)


# MANAGER → AGENT

graph_builder.add_conditional_edges(

    "manager",

    route_task,

    {
        "planner": "planner",

        "teacher": "teacher",

        "quiz": "quiz",

        "evaluator": "evaluator",
    }
)


# AGENTS → END

graph_builder.add_edge(
    "planner",
    END
)

graph_builder.add_edge(
    "teacher",
    END
)

graph_builder.add_edge(
    "quiz",
    END
)

graph_builder.add_edge(
    "evaluator",
    END
)


# Compile

study_graph = graph_builder.compile()


# ============================================================
# 13. MAIN AGENT FUNCTION
# ============================================================

def run_study_agent(user_message):

    if not user_message:

        return "Please enter a question."

    if not isinstance(
        user_message,
        str
    ):

        user_message = str(
            user_message
        )

    if not user_message.strip():

        return "Please enter a question."

    try:

        result = study_graph.invoke(
            {
                "user_request":
                user_message.strip()
            }
        )

        response = result.get(
            "response",
            "No response generated."
        )

        return content_to_text(
            response
        )

    except Exception as e:

        return (
            "### ❌ Error\n\n"
            "```text\n"
            + str(e)
            + "\n```"
        )


# ============================================================
# 14. GRADIO UI
# ============================================================

with gr.Blocks(
    title="AI Study Assistant Agent"
) as app:

    gr.Markdown(
        """
# 🎓 AI Study Assistant Agent

### Powered by Gemma + LangChain + LangGraph

Your personal AI assistant for:

📚 Learning  
📅 Study Planning  
📝 Quiz Generation  
✅ Answer Evaluation
"""
    )


    # ========================================================
    # AI ASSISTANT
    # ========================================================

    with gr.Tab(
        "🤖 AI Assistant"
    ):

        user_input = gr.Textbox(

            label="Ask your Study Assistant",

            placeholder=(
                "Example: Explain normalization "
                "in DBMS in simple words."
            ),

            lines=5
        )


        ask_button = gr.Button(

            "🚀 Ask AI Agent",

            variant="primary"
        )


        agent_output = gr.Markdown()


        ask_button.click(

            fn=run_study_agent,

            inputs=user_input,

            outputs=agent_output
        )


    # ========================================================
    # LEARN TOPIC
    # ========================================================

    with gr.Tab(
        "📚 Learn Topic"
    ):

        topic_input = gr.Textbox(

            label="Topic",

            placeholder=(
                "Example: Explain hypothesis testing"
            ),

            lines=3
        )


        topic_button = gr.Button(

            "📖 Explain Topic",

            variant="primary"
        )


        topic_output = gr.Markdown()


        def learn_topic(topic):

            return run_study_agent(

                "Explain this topic: "
                + str(topic)
            )


        topic_button.click(

            fn=learn_topic,

            inputs=topic_input,

            outputs=topic_output
        )


    # ========================================================
    # STUDY PLANNER
    # ========================================================

    with gr.Tab(
        "📅 Study Planner"
    ):

        plan_input = gr.Textbox(

            label="Study Requirements",

            placeholder=(
                "Example: I have a DBMS exam "
                "in 5 days and can study 4 hours per day."
            ),

            lines=5
        )


        plan_button = gr.Button(

            "📅 Create Study Plan",

            variant="primary"
        )


        plan_output = gr.Markdown()


        def create_plan(details):

            return run_study_agent(

                "Create a study plan: "
                + str(details)
            )


        plan_button.click(

            fn=create_plan,

            inputs=plan_input,

            outputs=plan_output
        )


    # ========================================================
    # QUIZ GENERATOR
    # ========================================================

    with gr.Tab(
        "📝 Quiz Generator"
    ):

        quiz_input = gr.Textbox(

            label="Quiz Topic",

            placeholder=(
                "Example: Statistics"
            ),

            lines=3
        )


        quiz_button = gr.Button(

            "📝 Generate Quiz",

            variant="primary"
        )


        quiz_output = gr.Markdown()


        def create_quiz(topic):

            return run_study_agent(

                "Create a quiz about "
                + str(topic)
            )


        quiz_button.click(

            fn=create_quiz,

            inputs=quiz_input,

            outputs=quiz_output
        )


    # ========================================================
    # ANSWER EVALUATOR
    # ========================================================

    with gr.Tab(
        "✅ Answer Evaluator"
    ):

        question_input = gr.Textbox(

            label="Question",

            placeholder=(
                "Enter your exam question"
            ),

            lines=3
        )


        answer_input = gr.Textbox(

            label="Your Answer",

            placeholder=(
                "Enter your answer"
            ),

            lines=8
        )


        evaluate_button = gr.Button(

            "✅ Evaluate My Answer",

            variant="primary"
        )


        evaluation_output = gr.Markdown()


        def evaluate_answer(
            question,
            answer
        ):

            request = f"""
Evaluate my answer.

Question:
{question}

Student Answer:
{answer}
"""

            return run_study_agent(
                request
            )


        evaluate_button.click(

            fn=evaluate_answer,

            inputs=[
                question_input,
                answer_input
            ],

            outputs=evaluation_output
        )


# ============================================================
# 15. START APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "7860"
        )
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=port,

        share=False
    )
