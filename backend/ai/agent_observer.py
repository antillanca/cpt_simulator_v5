"""Agent Observer (i@) - The guardian and executor.

Integrates Anchor Laws, Intent Classification, and the Student Engine.
"""
import logging
from typing import Dict, Any

from backend.ai.leyes_ancla import check_intent_violation
from backend.ai.intent_classifier import classify_intent
from backend.ai.student_engine import student
from backend.validation.validator import validator
from backend.core.orchestrator import orchestrator

logger = logging.getLogger(__name__)

class AgentObserver:
    """The i@ Agent: Observes user intent and ensures physical grounding."""

    async def process_request(self, user_text: str) -> Dict[str, Any]:
        """Process a natural language request from the user.
        
        Flow:
        1. Anchor Law Check (Fast)
        2. Question Detection (Onda Mode)
        3. Intent Classification (Action Mode)
        4. Execution/Generation
        """
        logger.info(f"[i@] Processing user request: '{user_text}'")

        # 1. Check for Anchor Law Violations
        violation_msg = check_intent_violation(user_text)
        if violation_msg:
            logger.warning(f"[i@] Anchor Law Violation: {violation_msg}")
            return {"status": "blocked", "message": violation_msg}

        # 2. Detect if it's a question (Onda Mode)
        if self._is_question(user_text):
            logger.info("[i@] Onda Mode: Answering question using Knowledge Tree RAG.")
            answer = await self._answer_question(user_text)
            return {
                "status": "success",
                "agent": "i@",
                "mode": "Onda (RAG)",
                "message": answer
            }

        # 3. Classify Intent (Action Mode)
        module_name = classify_intent(user_text)
        
        lua_code = None
        if module_name:
            # Try to retrieve confirmed knowledge
            module = student.get_module(module_name)
            if module and module.get("status") == "confirmed" and module.get("lua_code"):
                logger.info(f"[i@] Knowledge Retrieval: Using existing '{module_name}' layer.")
                lua_code = module["lua_code"]

        # 4. If no code found, use Student Engine to generate
        if not lua_code:
            logger.info("[i@] Generation Mode: Student Engine is thinking...")
            target_name = module_name or "adhoc_request"
            lua_code = await student.generate_lua(target_name, description=user_text)

        if not lua_code:
            return {"status": "error", "message": "The i@ agent could not formulate a physical rule."}

        # 5. Security Validation
        is_safe, error = validator.validate_rule(lua_code)
        if not is_safe:
            return {"status": "error", "message": f"Generated code failed security check: {error}"}

        # 6. Success: Return the code for injection
        return {
            "status": "success",
            "agent": "i@",
            "intent": module_name or "custom",
            "lua_code": lua_code,
            "message": "i@ has verified and generated the rule."
        }

    def _is_question(self, text: str) -> bool:
        """Detect if the input is a request for information (question) vs action."""
        question_words = [
            "qué es", "que es", "cómo funciona", "como funciona", "explica",
            "dime", "qué significa", "what is", "how does", "explain", "tell me"
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in question_words) or "?" in text

    async def _answer_question(self, question: str) -> str:
        """Answer a question using RAG over confirmed Knowledge Tree layers."""
        from backend.ai.student_engine import ollama_generate
        
        # Collect context from confirmed modules
        layers = student.get_confirmed_layers()
        knowledge_summary = ""
        for name, mod in layers:
            knowledge_summary += f"- {name}: {mod['description']}\n"

        if not knowledge_summary:
            knowledge_summary = "No knowledge layers confirmed yet."

        prompt = f"""You are the i@ Observer Agent. 
You answer questions based ONLY on the following knowledge tree of confirmed physical/mathematical layers:
{knowledge_summary}

User Question: {question}

Instructions:
- Be concise.
- Use the confirmed knowledge to explain.
- If the topic is not in the knowledge tree, explain that you are still learning it.
- Answer in the same language as the question.

Response:"""
        
        response = await ollama_generate(prompt)
        return response or "I am currently unable to process that question."

# Singleton
agent_i_at = AgentObserver()
