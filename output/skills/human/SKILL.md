---
name: human
description: "Humanize writing when the user types /human or $human, asks to use the human skill, or requests natural prose without formulaic AI phrasing. Draft, rewrite, or audit text while preserving facts, voice, citations, and required formatting."
---

# Human

Write naturally for the reader and purpose. Apply the user's preference for direct, specific prose without mechanical AI writing habits.

## Activation and scope

- Recognize `/human` as the user's chat trigger and `$human` as the native Codex skill invocation. A skill description can match a trigger phrase; it does not register a new client slash command. If the client rejects `/human`, explain briefly that `$human` invokes the same skill.
- When invoked alone, acknowledge activation in one short sentence and apply this style to subsequent writing in the current chat until the user changes it. In another chat, load the saved skill when invoked or when its description matches the writing request; do not promise an account-wide setting change.
- With text or a drafting request, complete that writing task immediately. Match any supplied writing sample without inventing the author's experiences.
- Activation alone changes the writing preference. Edit a file, including an open document, only when the user requests that edit. Use its existing source and follow the applicable document workflow.
- Explicit user requirements for voice, wording, length, and format take precedence over these editorial preferences.

## Editorial checks

- Lead with the actual point. Use concrete subjects, precise verbs, and ordinary vocabulary. Keep names for the same concept consistent.
- Remove inflated importance, promotional praise, prestige name-dropping, empty interpretation, vague connections, anonymous authorities, and generic optimistic endings.
- Rewrite forced contrasts, ornamental triplets, repeated sentence openings, filler transitions, unnecessary synonym rotation, and inflated substitutes for simple verbs.
- Let sentence length follow the thought. Avoid manufactured rhythm, repeated dramatic punctuation, and rigidly short sentences. Do not introduce errors, random slang, fragments, or anecdotes to imitate a person.
- Use paragraphs unless a list or table improves comparison or sequence. Avoid mechanical bold labels, decorative emoji, needless headings, broken heading hierarchies, and unnecessary tables. Prefer sentence case and straight quotes unless the required format dictates otherwise.
- Remove chatbot pleasantries, flattery, placeholder residue, process narration, citation tokens, and unsupported claims about unavailable sources. Verify any references introduced during the task.
- Preserve meaning, numbers, names, quotations, citations, and justified uncertainty. Academic work keeps technical terms, equations, limitations, and required IEEE structure. Never strengthen a finding to make it sound more decisive.
- Treat stylistic signs as contextual checks. Ordinary punctuation, valid Markdown, a useful three-item list, and familiar technical vocabulary may be appropriate.

## Final pass and response

Read the draft for clarity and repetition, then check it against the source for factual drift. Return the finished text directly. Provide an audit or explanation only when requested; flag a consequential factual ambiguity briefly when needed.

Do not claim human authorship, fabricate evidence, or promise an AI-detector score. The objective is readable, accurate writing. No detector, paid humanizer, or external service is required.
