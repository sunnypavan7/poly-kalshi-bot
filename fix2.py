from backend.config import settings, CITIES 
content = open('main.py').read() 
content = content.replace('from backend.config import PAPER_TRADING, CITIES', 'from backend.config import settings, CITIES') 
content = content.replace('PAPER_TRADING', 'settings.PAPER_TRADING') 
content = content.replace('SCAN_INTERVAL_SEC', 'settings.SCAN_INTERVAL_SEC') 
open('main.py', 'w').write(content) 
