"""
Natural Conversation Handling System
Provides intent classification, greeting detection, small talk, gratitude,
acknowledgments, apologies, farewells, identity, capabilities, and mixed-intent routing.
Multilingual support for English, Bangla, and Banglish.
Bypasses heavy vector search, BM25Plus, and LLM calls for pure conversational messages.
"""

import re
import unicodedata
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("code_rag.rag.conversation")


class ConversationHandler:
    def __init__(self):
        # Repeated character collapsing
        self.repeat_char_re = re.compile(r'([a-zA-Z])\1{2,}')

        # Action verbs indicating technical commands
        self.action_verbs = {
            "explain", "show", "find", "implement", "write", "create", "build",
            "debug", "fix", "analyze", "trace", "search", "lookup", "give", "display",
            "list", "get", "print", "extract", "test", "run", "refactor"
        }

        # Code keywords and technical indicators
        self.code_indicators = {
            "function", "functions", "class", "classes", "def ", "method", "methods",
            "()", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".md",
            "component", "components", "frontend", "backend", "feature", "features",
            "implement", "implementation", "decorator", "decorators", "variable",
            "variables", "api", "endpoint", "endpoints", "route", "routes", "retriever",
            "retrieval", "hybrid", "rag", "database", "sql", "sqlite", "vector",
            "chroma", "bm25", "repository", "repo", "import", "export", "algorithm",
            "complexity", "async", "await", "loop", "syntax", "code", "file", "files",
            "tree", "directory", "pipeline", "service", "model", "token", "schema"
        }

    def normalize(self, text: str) -> str:
        """
        Normalizes text:
        1. Unicode NFKC normalization
        2. Normalizes quotes and apostrophes
        3. Strips emojis for intent matching
        4. Collapses excessive repeated characters (e.g. 'Hiiii' -> 'Hi', 'Heyyyyy' -> 'Hey')
        5. Normalizes whitespace
        """
        if not text:
            return ""
        t = unicodedata.normalize('NFKC', text)
        t = re.sub(r"[’‘´`]", "'", t)
        t = re.sub(r'[“”]', '"', t)
        # Remove emojis safely
        t = re.sub(r'[\U00010000-\U0010ffff]', ' ', t)
        # Targeted word collapses
        t = re.sub(r'(?i)\b(h)i{2,}\b', r'\1i', t)
        t = re.sub(r'(?i)\b(he)y{2,}\b', r'\1y', t)
        t = re.sub(r'(?i)\b(hell)o{2,}\b', r'\1o', t)
        t = re.sub(r'(?i)\b(thank)s{2,}\b', r'\1s', t)
        # General character collapse (3 or more down to 2)
        t = self.repeat_char_re.sub(r'\1\1', t)
        return re.sub(r'\s+', ' ', t).strip()

    def detect_language(self, text: str) -> str:
        """Detects whether text is in Bengali script, Banglish, or English."""
        # 1. Bengali Unicode block \u0980-\u09FF
        if re.search(r'[\u0980-\u09FF]', text):
            return "bangla"

        # 2. Banglish common tokens
        banglish_markers = {
            "kemon", "achen", "asen", "aso", "acho", "achis", "bhalo", "valo",
            "dhonnobad", "thik", "ache", "ase", "apni", "tumi", "tui", "amake",
            "amader", "khobor", "shuvo", "ratri", "biday", "ki", "korcho",
            "korchen", "koris", "onek", "dhonnobad.", "bolben", "lagle",
            "pari", "paro", "hobe", "chai", "dao", "dekhao", "bhai", "somossa",
            "nei", "shagotom", "bujhechi"
        }
        tokens = set(re.findall(r'[a-zA-Z]+', text.lower()))
        if tokens.intersection(banglish_markers):
            return "banglish"

        return "english"

    def is_explicit_technical(self, text: str) -> bool:
        """
        Identifies whether a message is an explicit programming or code request.
        Does NOT treat isolated generic words (e.g. 'notification', 'feature', 'service')
        as technical proof unless placed in meaningful programming context.
        """
        t_lower = text.lower().strip()

        # 1. Code blocks or function calls like greet_user() or func()
        if re.search(r'\b[a-zA-Z_]\w*\s*\(\s*\)', text):
            return True
        if re.search(r'`[^`]+`', text):
            return True
        if re.search(r'\b[a-zA-Z0-9_\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|sql|yaml|yml)\b', t_lower):
            return True

        # 2. Programming comparisons e.g. "difference between Hi and Hello in Python"
        if "difference between" in t_lower and any(lang in t_lower for lang in ["python", "java", "javascript", "c++", "go", "rust"]):
            return True

        # 3. Action verbs on code entities
        action_verbs = r'(?:explain|show|find|implement|write|create|build|debug|fix|analyze|trace|search|lookup|give|display|refactor)'
        code_entities = r'(?:function|class|method|component|feature|implementation|module|service|algorithm|endpoint|api|decorator|decorators)'
        
        # e.g. "Explain the hello function in Python", "Find the greet_user() function"
        if re.search(rf'\b{action_verbs}\b.*\b{code_entities}\b', t_lower):
            # Guard against gratitude expressions like "Thanks for your help with the feature"
            if not re.search(r'\b(?:thanks|thank you)\b', t_lower):
                return True

        # e.g. "Show me the hello.py file", "Explain the notification service in my project"
        if re.search(r'\b(?:in my project|in this project|in our repo|in the repo|in my repo|in the codebase|in python)\b', t_lower):
            if any(w in t_lower for w in ["function", "service", "component", "file", "decorator", "feature", "endpoint"]):
                return True

        return False

    def is_clause_conversational(self, clause: str) -> Optional[str]:
        """
        Evaluates a single normalized clause to determine if it belongs to a
        conversational category: gratitude, acknowledgment, apology, greeting,
        small talk, farewell, identity, or capabilities.
        """
        c = re.sub(r"^[^\w\s\u0980-\u09FF']+|[^\w\s\u0980-\u09FF']+$", '', clause).strip().lower()
        if not c:
            return None

        # 1. Gratitude
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"thanks", r"thank you", r"thank you so much", r"thanks a lot", r"many thanks",
            r"much appreciated", r"i appreciate it", r"i really appreciate your help",
            r"thanks for your help", r"thanks for the help", r"thanks for the notification",
            r"that's helpful,? thanks", r"that's helpful", r"helpful thanks",
            r"dhonnobad", r"onek dhonnobad", r"apnake onek dhonnobad",
            r"ধন্যবাদ", r"অনেক ধন্যবাদ", r"আপনাকে অনেক ধন্যবাদ"
        ]):
            return "gratitude"

        # 2. Acknowledgment
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"okay", r"ok", r"got it", r"understood", r"perfect", r"great", r"excellent",
            r"sounds good", r"alright", r"cool", r"nice", r"awesome", r"wonderful", r"fine",
            r"no worries", r"you're welcome", r"you are welcome",
            r"thik ache", r"tik ase", r"accha", r"bujhlam", r"thik ache bhai",
            r"ঠিক আছে", r"আচ্ছা", r"বুঝেছি", r"বুঝলাম"
        ]):
            return "acknowledgment"

        # 3. Apology
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"sorry", r"i'm sorry", r"i am sorry", r"my mistake", r"my bad", r"sorry about that",
            r"apologies", r"excuse me", r"that's my fault",
            r"kono somossa nei", r"somossa nei", r"দুঃখিত", r"ক্ষমা করবেন", r"কোনো সমস্যা নেই"
        ]):
            return "apology"

        # 4. Greeting
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"hi", r"hello", r"hey", r"hey there", r"hi there", r"good morning", r"good afternoon",
            r"good evening", r"what's up", r"whats up", r"howdy", r"assalamu alaikum", r"salam",
            r"আসসালামু আলাইকুম", r"সালাম"
        ]):
            return "greeting"

        # 5. Small talk
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"how are you", r"how are you doing", r"what's going on", r"whats going on",
            r"how is your day", r"how's your day", r"are you there", r"what are you doing",
            r"nice to meet you", r"hope you're doing well", r"can we talk", r"are you available",
            r"apni kemon achen", r"kemon asen", r"kemon acho", r"sob bhalo", r"ki khobor", r"bhalo achen",
            r"তুমি কেমন আছো", r"কেমন আছেন", r"কি খবর", r"সব ভালো", r"ভালো আছি", r"ভালো আছেন", r"আপনি কেমন আছেন"
        ]):
            return "small_talk"

        # 6. Farewell
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"bye", r"goodbye", r"see you", r"see you later", r"take care", r"have a nice day",
            r"good night", r"catch you later", r"biday", r"shuvo ratri", r"bhalo thakben",
            r"বিদায়", r"শুভ রাত্রি", r"ভালো থাকবেন"
        ]):
            return "farewell"

        # 7. Identity
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"who are you", r"what are you", r"are you an ai", r"what is your name", r"tell me about yourself",
            r"apni ke", r"tumi ke", r"tomar nam ki", r"আপনি কে", r"তুমি কে", r"তোমার নাম কি"
        ]):
            return "identity"

        # 8. Capabilities
        if any(re.search(rf'^{p}$', c) or re.search(p, c) for p in [
            r"what can you do", r"how can you help me", r"what are your features", r"what are your capabilities",
            r"can you help me", r"what kind of questions can i ask",
            r"apni ki korte paren", r"tumi ki korte paro", r"আপনি কি করতে পারেন", r"তুমি কি করতে পারো", r"কি কি সুবিধা আছে"
        ]):
            return "capabilities"

        return None

    def detect_mixed_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects if query starts with a conversational greeting, gratitude, or acknowledgment
        followed by a substantive technical request.
        Examples:
          - 'Hi, explain graph.py.'
          - 'Hi explain graph.py.'
          - 'Hello, how does hybrid retrieval work?'
          - 'Thanks, show me retriever.py.'
          - 'Good morning, explain Python decorators.'
          - 'Perfect, now give me all project files one by one.'
          - 'Hello, explain analyze_query().'
        """
        norm = self.normalize(text)
        prefix_pattern = (
            r'^(?P<prefix>hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|'
            r'assalamu alaikum|salam|আসসালামু আলাইকুম|সালাম|'
            r'thanks|thank you|thank you so much|ধন্যবাদ|অনেক ধন্যবাদ|dhonnobad|onek dhonnobad|'
            r'perfect|great|excellent|okay|ok|thik ache|ঠিক আছে)'
            r'(?:[,.\-!:\n\s]+|\s+)'
            r'(?P<rest>.+)$'
        )
        m = re.match(prefix_pattern, norm, flags=re.IGNORECASE)
        if not m:
            return None

        prefix_raw = m.group("prefix").strip()
        rest = m.group("rest").strip()

        # If the rest is ALSO conversational (e.g. 'Hi, how are you?', 'Okay, got it'),
        # this is composite conversation, NOT mixed intent!
        if self.is_clause_conversational(rest):
            return None

        cleaned_rest = re.sub(r'^(?:now\s+|can you\s+|please\s+|amake\s+)+', '', rest, flags=re.IGNORECASE).strip()
        cleaned_lower = cleaned_rest.lower()

        # Is the rest substantive technical?
        has_tech = (
            self.is_explicit_technical(cleaned_rest) or
            any(w in cleaned_lower for w in [
                "explain", "show", "find", "implement", "how does", "what is", "why",
                "all code", "all files", "give me", "break down", "line by line", "retrieval",
                "hybrid", "decorator", "decorators", "graph.py", "retriever.py", "analyze_query"
            ])
        )
        if not has_tech:
            return None

        lang = self.detect_language(text)
        prefix_cat = self.is_clause_conversational(prefix_raw) or "greeting"

        if prefix_cat == "gratitude":
            conv_prefix = "You're welcome! " if lang == "english" else ("আপনাকে স্বাগতম! " if lang == "bangla" else "Welcome! ")
        elif prefix_cat == "acknowledgment":
            conv_prefix = "Understood! " if lang == "english" else ("ঠিক আছে! " if lang == "bangla" else "Thik ache! ")
        else:
            conv_prefix = "Hello! " if lang == "english" else ("হ্যালো! " if lang == "bangla" else "Hello! ")

        return {
            "type": "mixed_intent",
            "is_conversational": True,
            "is_mixed": True,
            "category": f"{prefix_cat}_mixed",
            "conversational_prefix": conv_prefix,
            "technical_question": rest,
            "cleaned_technical_question": cleaned_rest,
            "language": lang
        }

    def detect_pure_conversational(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Matches single or composite pure conversational messages.
        Returns intent dictionary if conversational, or None if technical.
        """
        norm = self.normalize(text)
        t_clean = re.sub(r"^[^\w\s\u0980-\u09FF']+|[^\w\s\u0980-\u09FF']+$", '', norm).strip().lower()
        lang = self.detect_language(text)

        # 1. Check if the entire string matches a known conversational clause directly
        direct_cat = self.is_clause_conversational(t_clean)
        if direct_cat:
            return {
                "category": direct_cat,
                "language": lang,
                "response": self.generate_response(t_clean, direct_cat, lang)
            }

        # 2. Split message into clauses by punctuation: , . ! ? ; \n
        clauses = [c.strip() for c in re.split(r'[,.!?;\n]+', norm) if c.strip()]
        if len(clauses) > 1:
            clause_cats = [self.is_clause_conversational(c) for c in clauses]
            if all(cat is not None for cat in clause_cats):
                primary_cat = "small_talk"
                if "gratitude" in clause_cats:
                    primary_cat = "gratitude"
                elif "apology" in clause_cats:
                    primary_cat = "apology"
                elif "acknowledgment" in clause_cats and "greeting" not in clause_cats:
                    primary_cat = "acknowledgment"
                elif "greeting" in clause_cats:
                    primary_cat = "greeting"
                elif "farewell" in clause_cats:
                    primary_cat = "farewell"
                elif "identity" in clause_cats:
                    primary_cat = "identity"
                elif "capabilities" in clause_cats:
                    primary_cat = "capabilities"

                return {
                    "category": primary_cat,
                    "language": lang,
                    "response": self.generate_response(t_clean, primary_cat, lang)
                }

        return None

    def detect_intent(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        active_session: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Master classifier returning intent evaluation, routing target, and direct response
        when the message is pure conversational.
        """
        norm = self.normalize(question)
        lang = self.detect_language(question)
        t_clean = re.sub(r"^[^\w\s\u0980-\u09FF']+|[^\w\s\u0980-\u09FF']+$", '', norm).strip().lower()

        # 1. Active workflow and export continuation
        has_active_export = bool(active_session and active_session.get("pending_files"))
        is_all_files_phrase = t_clean in {
            "all file", "all files", "export all", "export all files", "export all remaining files",
            "all remaining files", "all remaining code", "give all remaining files", "give all remaining file",
            "give all files", "all remaining", "give remaining", "baki shob file", "baki shob code",
            "shob code dao", "shob file dao"
        }
        if t_clean in {"next", "continue", "পরেরটা", "পরের কোড দাও", "baki code dao"} or is_all_files_phrase:
            if has_active_export:
                return {
                    "type": "export_continuation",
                    "is_conversational": False,
                    "is_contextual": True,
                    "workflow": "repo_code_export"
                }
            elif t_clean in {"continue", "next"}:
                # Scenario 51: User says Continue/Next with NO active workflow
                return {
                    "type": "pure_conversation",
                    "is_conversational": True,
                    "is_pure": True,
                    "category": "context_continuation_prompt",
                    "language": lang,
                    "response": "What would you like me to continue with? Please specify a file, function, or question."
                }

        # 2. Context follow-up checks (with history)
        has_history = bool(history and len(history) > 0)
        is_followup = (
            t_clean in {"why", "why?", "explain more", "elaborate", "karon ki", "কেন?", "কেন", "tell me more"} or
            (has_history and (
                re.search(r'\b(?:what about|how about|and what about)\b', t_clean) or
                re.search(r'\b(?:its dependencies|its|this|that|the previous one|same file|same function)\b', t_clean)
            ))
        )
        if is_followup:
            if has_history:
                last_asst = next((m for m in reversed(history) if m.get("role") == "assistant" and m.get("content")), None)
                last_user = next((m for m in reversed(history) if m.get("role") == "user" and m.get("content")), None)
                context_str = (last_user.get("content") if last_user else "") or (last_asst.get("content", "")[:200] if last_asst else "")
                return {
                    "type": "context_followup",
                    "is_conversational": False,
                    "is_contextual": True,
                    "workflow": "contextual_explanation",
                    "context_query": f"{context_str} - {question}"
                }
            else:
                # Scenario 52: User says "Why?" with no context
                return {
                    "type": "pure_conversation",
                    "is_conversational": True,
                    "is_pure": True,
                    "category": "context_clarification",
                    "language": lang,
                    "response": "Could you please clarify what you would like to know more about? Feel free to ask about any specific function, file, or concept."
                }

        # Context confirmation: "Yes", "Yeah", "Sure"
        if t_clean in {"yes", "yeah", "yup", "ha", "sure", "proceed", "হ্যাঁ", "হাঁ"}:
            pending_confirmation = False
            if has_history:
                last_asst = next((m for m in reversed(history) if m.get("role") == "assistant" and m.get("content")), None)
                if last_asst and ("?" in last_asst.get("content", "") or any(w in last_asst.get("content", "").lower() for w in ["proceed", "would you like", "should i"])):
                    pending_confirmation = True

            if pending_confirmation:
                return {
                    "type": "context_confirmation",
                    "is_conversational": False,
                    "is_contextual": True,
                    "workflow": "confirmation_response"
                }
            else:
                # Scenario 50: User says "Yes" with no pending confirmation
                return {
                    "type": "pure_conversation",
                    "is_conversational": True,
                    "is_pure": True,
                    "category": "acknowledgment",
                    "language": lang,
                    "response": "Understood! How can I help you today?"
                }

        # 3. Explicit technical requests
        if self.is_explicit_technical(question) and not self.detect_mixed_intent(question):
            return {
                "type": "technical_request",
                "is_conversational": False,
                "reason": "Explicit technical programming request"
            }

        # 4. Mixed intent check
        mixed = self.detect_mixed_intent(question)
        if mixed:
            return mixed

        # 5. Pure conversational check (single or composite)
        pure = self.detect_pure_conversational(question)
        if pure:
            return {
                "type": "pure_conversation",
                "is_conversational": True,
                "is_pure": True,
                **pure
            }

        # 6. Default: technical request
        return {
            "type": "technical_request",
            "is_conversational": False
        }

    def generate_response(self, text: str, category: str, lang: str = "english") -> str:
        """Generates natural, high quality, concise responses for conversational intents."""
        if lang == "bangla":
            if category == "gratitude":
                return "আপনাকে স্বাগতম! আরও কোনো সাহায্য লাগলে জানাবেন।"
            elif category == "acknowledgment":
                return "ঠিক আছে! আপনার পরবর্তী নির্দেশনার জন্য অপেক্ষা করছি।"
            elif category == "apology":
                return "কোনো সমস্যা নেই! আমি আপনাকে সাহায্য করতে প্রস্তুত।"
            elif category == "greeting":
                if "আসসালামু" in text or "সালাম" in text:
                    return "ওয়ালাইকুম আসসালাম! আপনাকে কীভাবে সাহায্য করতে পারি?"
                return "হ্যালো! আশা করি ভালো আছেন। আজ আপনাকে কীভাবে সাহায্য করতে পারি?"
            elif category == "small_talk":
                return "আমি ভালো আছি, ধন্যবাদ! আপনাকে কীভাবে সাহায্য করতে পারি?"
            elif category == "farewell":
                return "বিদায়! ভালো থাকবেন এবং শুভ কোডিং!"
            elif category == "identity":
                return "আমি আপনার গিটহাব রিপোজিটরি কোড অ্যাসিস্ট্যান্ট এবং এআই প্রোগ্রামিং পার্টনার। আমি কোডবেস বিশ্লেষণ, কোড ব্যাখ্যা এবং সফটওয়্যার ইঞ্জিনিয়ারিং সম্পর্কিত প্রশ্নের উত্তর দিতে সাহায্য করি।"
            elif category == "capabilities":
                return "আমি গিটহাব রিপোজিটরি কোড বিশ্লেষণ, লাইন-বাই-লাইন ব্যাখ্যা এবং সম্পূর্ণ কোড এক্সপোর্টে সাহায্য করতে পারি।"
            return "হ্যালো! কীভাবে সাহায্য করতে পারি?"
        elif lang == "banglish":
            if category == "gratitude":
                return "Apnake shagotom! Ar kono help lagle janaben."
            elif category == "acknowledgment":
                return "Thik ache! Porer instruction er jonno ami ready."
            elif category == "apology":
                return "Kono somossa nei! Ami apnake help korte ready."
            elif category == "greeting":
                if "salam" in text or "assalamu" in text:
                    return "Wa Alaikum Assalam! Apnake kivabe help korte pari?"
                return "Hello! Apnake kivabe help korte pari?"
            elif category == "small_talk":
                return "Ami bhalo achi, dhonnobad! Apnake kivabe help korte pari?"
            elif category == "farewell":
                return "Biday! Bhalo thakben ar happy coding!"
            elif category == "identity":
                return "Ami apnar GitHub Repository Code Assistant and AI programming partner."
            elif category == "capabilities":
                return "Ami repository codebase explore, line-by-line explanation, ar complete code export korte pari."
            return "Hello! Kivabe help korte pari?"
        else:
            # English
            if category == "gratitude":
                if "notification" in text:
                    return "You're very welcome! Let me know if you need any further updates or assistance."
                return "You're very welcome! Glad I could help."
            elif category == "acknowledgment":
                return "Great! Let me know if you need anything else."
            elif category == "apology":
                return "No worries at all!"
            elif category == "greeting":
                if "good morning" in text:
                    return "Good morning! Hope you're having a great day. How can I help you?"
                elif "good afternoon" in text:
                    return "Good afternoon! How can I assist you today?"
                elif "good evening" in text:
                    return "Good evening! How can I assist you today?"
                elif "what's up" in text or "whats up" in text:
                    return "Not much! I'm ready to help with your code and questions. What would you like to work on?"
                elif "hey" in text:
                    return "Hey! How can I assist you today?"
                return "Hello! How can I help you today?"
            elif category == "small_talk":
                if "are you there" in text:
                    return "Yes, I'm here and ready to help!"
                elif "nice to meet you" in text:
                    return "Nice to meet you too! How can I help you today?"
                return "I'm doing well, thank you for asking! How can I help you today?"
            elif category == "farewell":
                if "good night" in text:
                    return "Good night! Rest well and see you next time!"
                return "Goodbye! Have a great day and happy coding!"
            elif category == "identity":
                return "I am your GitHub Repository Code Assistant and AI Programming Partner. I can help you explore, analyze, and understand codebases, explain source code, and answer software engineering questions."
            elif category == "capabilities":
                return "I can help you analyze GitHub repositories, provide line-by-line code breakdowns, export complete repository code, and answer software engineering questions."
            return "Hello! How can I help you today?"

    def handle_greeting(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "greeting", lang)

    def handle_small_talk(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "small_talk", lang)

    def handle_gratitude(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "gratitude", lang)

    def handle_acknowledgment(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "acknowledgment", lang)

    def handle_apology(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "apology", lang)

    def handle_farewell(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "farewell", lang)

    def handle_identity(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "identity", lang)

    def handle_capabilities(self, text: str, lang: str = "english") -> str:
        return self.generate_response(text, "capabilities", lang)

    # Backward-compatible alias
    evaluate = detect_intent
