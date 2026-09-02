# i18n.py
import locale
import gettext

lang_code = locale.getlocale()[0] or 'en_EN.UTF-8'
lang_code = lang_code.split('_')[0] # fr_FR => fr

lang = gettext.translation("gutcheck", localedir="locales", languages=[lang_code], fallback=True)
_ = lang.gettext
n_ = lang.ngettext
