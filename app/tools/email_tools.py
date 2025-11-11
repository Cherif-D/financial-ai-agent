# Outil pour l'envoi d'emails via SMTP

# app/tools/email_tools.py
import os
import re
import ssl
import smtplib
from email.message import EmailMessage
from typing import Tuple, Optional

from email_validator import validate_email, EmailNotValidError
from langchain.tools import Tool

# ---------- Utils ----------
def _sanitize(s: str) -> str:
    s = s.replace("\r", " ").strip()
    s = re.sub(r"[“”’]", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_keyvals(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse un format simple:
        to: destinataire@example.com
        subject: Objet
        body: Corps de l'email ...
    -> renvoie (to, subject, body)
    """
    to = subject = body = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    buf_body = []
    in_body = False

    for ln in lines:
        low = ln.lower()
        if low.startswith("to:") or low.startswith("à:"):
            to = ln.split(":", 1)[1].strip()
            in_body = False
        elif low.startswith("subject:") or low.startswith("objet:"):
            subject = ln.split(":", 1)[1].strip()
            in_body = False
        elif low.startswith("body:") or low.startswith("corps:"):
            body = ln.split(":", 1)[1].strip()
            in_body = True
        else:
            if in_body:
                buf_body.append(ln)

    if in_body and buf_body:
        body = (body + "\n" + "\n".join(buf_body)).strip() if body else "\n".join(buf_body).strip()

    return to, subject, body

def _validate_to(addr: str) -> str:
    try:
        v = validate_email(addr, check_deliverability=False)
        return v.normalized
    except EmailNotValidError as e:
        raise ValueError(f"Adresse e-mail invalide: {e}")

def _smtp_conf() -> Tuple[str, int, str, str, str, bool]:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    from_addr = os.getenv("SMTP_FROM", user or "")
    use_tls = os.getenv("SMTP_TLS", "true").lower() in {"1","true","yes","on"}
    if not (host and port and user and pwd and from_addr):
        raise RuntimeError("Config SMTP manquante (SMTP_HOST/PORT/USER/PASS/FROM).")
    return host, port, user, pwd, from_addr, use_tls

# ---------- Tool: Brouillon ----------
def _draft_email_fn(query: str) -> str:
    """
    Rédige un brouillon d'email professionnel en FR.
    Entrée libre ou en keyvals:
    to: ...
    subject: ...
    body: ...
    """
    q = _sanitize(query)
    to, subject, body = _parse_keyvals(q)

    # Ton pro et clair si pas de body fourni
    if not subject:
        subject = "Demande d'information"
    if not body:
        body = (
            "Bonjour,\n\n"
            "Je me permets de vous contacter au sujet de ...\n\n"
            "Pourriez-vous me confirmer ... ?\n\n"
            "Bien cordialement,\n"
            "Diallo Mamadou Cherif"
        )

    header = []
    if to:
        header.append(f"to: {to}")
    header.append(f"subject: {subject}")
    header.append("body: " + body)

    return "\n".join(header)

draft_email = Tool.from_function(
    func=_draft_email_fn,
    name="draft_email",
    description=(
        "Rédige un brouillon d'e-mail FR professionnel. "
        "Accepte du texte libre ou le format 'to:/subject:/body:'. "
        "Renvoie le triplet to/subject/body prêt pour l'envoi."
    ),
)

# ---------- Tool: Envoi SMTP ----------
def _send_email_smtp_fn(query: str) -> str:
    """
    Envoie un e-mail via SMTP (STARTTLS si SMTP_TLS=true).
    Entrée attendue AU FORMAT:
    to: destinataire@example.com
    subject: Objet
    body: Corps...
    """
    q = _sanitize(query)
    to, subject, body = _parse_keyvals(q)
    if not to or not subject or not body:
        return ("❌ Format manquant. Fourni:\n"
                f"to: {to}\nsubject: {subject}\nbody: {bool(body)}\n"
                "Exemple:\n"
                "to: contact@exemple.com\n"
                "subject: Candidature stage\n"
                "body: Bonjour, ...")

    to = _validate_to(to)
    host, port, user, pwd, from_addr, use_tls = _smtp_conf()

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    if use_tls and port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host=host, port=port, context=context) as server:
            server.login(user, pwd)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host=host, port=port) as server:
            server.ehlo()
            if use_tls:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            server.login(user, pwd)
            server.send_message(msg)

    return f"✅ E-mail envoyé à {to} (sujet: {subject})."

send_email_smtp = Tool.from_function(
    func=_send_email_smtp_fn,
    name="send_email_smtp",
    description=(
        "Envoie un e-mail via SMTP (config .env: SMTP_*). "
        "Entrée OBLIGATOIRE au format lignes 'to:/subject:/body:'."
    ),
)
