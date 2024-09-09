from lighteval.tasks.templates.translation_literals import TranslationLiterals


def decapitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].lower() + word[1:]


def capitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].upper() + word[1:]


def fix_ending_punct(ctx: str, translation_literals: TranslationLiterals):
    ctx = ctx.strip()
    if len(ctx) == 0:
        return ctx
    if ctx.endswith("?"):
        ctx = ctx[:-1] + translation_literals.question_mark
    elif ctx.endswith("."):
        ctx = ctx[:-1] + translation_literals.full_stop
    elif ctx.endswith(","):
        ctx = ctx[:-1] + translation_literals.comma
    elif ctx.endswith(":"):
        ctx = ctx[:-1] + translation_literals.colon
    return ctx


def is_ended_sentence(text: str, translation_literals: TranslationLiterals):
    return text.strip().endswith(f"{translation_literals.question_mark}{translation_literals.full_stop}{translation_literals.colon}")


def should_follow_sentence_space(prefix: str, translation_literals: TranslationLiterals):
    return prefix.strip().endswith(f"{translation_literals.question_mark}{translation_literals.full_stop}{translation_literals.colon}{translation_literals.comma}")


def fix_capitalization(prefix: str, text: str, translation_literals: TranslationLiterals):
    if len(prefix) == 0:
        return capitalize(text)
    
    if prefix.endswith("\n"):
        return capitalize(text)

    # TODO: Prob cache this
    return capitalize(text) if is_ended_sentence(prefix, translation_literals) else decapitalize(text)

