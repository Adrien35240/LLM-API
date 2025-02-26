from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import json
import httpx

app = FastAPI(title="Web Scraper API with Ollama")

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Web Scraper API avec intégration Ollama opérationnelle"}

@app.get("/search")
async def search(query: str):
    try:
        # Exemple avec une recherche simple sur DuckDuckGo
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        for result in soup.select(".result"):
            title_element = result.select_one(".result__title")
            link_element = result.select_one(".result__url")
            snippet_element = result.select_one(".result__snippet")
            
            if title_element and link_element:
                title = title_element.get_text(strip=True)
                link = link_element.get("href", "")
                snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                
                results.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet
                })
        
        return {"query": query, "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scrape")
async def scrape(url: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extraction basique
        title = soup.title.string if soup.title else ""
        
        # Extraction des paragraphes
        paragraphs = [p.get_text(strip=True) for p in soup.select("p")]
        
        # Extraction des liens
        links = [{"text": a.get_text(strip=True), "url": a.get("href")} 
                for a in soup.select("a") if a.get("href")]
        
        return {
            "url": url,
            "title": title,
            "content": {
                "paragraphs": paragraphs[:10],  # Limiter pour l'exemple
                "links": links[:10]  # Limiter pour l'exemple
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ollama/search")
async def ollama_search(query: str):
    try:
        # 1. Récupérer les résultats de recherche
        search_results = await search(query)
        
        # 2. Préparer le texte à résumer pour Ollama
        results_text = f"Résultats de recherche pour '{query}':\n\n"
        
        for idx, result in enumerate(search_results["results"][:5]):  # Limiter à 5 résultats pour performance
            results_text += f"{idx+1}. {result['title']}\n"
            results_text += f"   URL: {result['url']}\n"
            results_text += f"   {result['snippet']}\n\n"
        
        # 3. Préparer la requête pour Ollama avec instruction
        prompt = f"""Résume les informations suivantes en un paragraphe concis qui répond à la requête "{query}":

{results_text}

Ton résumé doit être informatif, objectif et contenir les points clés des résultats."""
        
        # 4. Appeler Ollama (API locale)
        async with httpx.AsyncClient(timeout=60.0) as client:
            ollama_response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "num_gpu": 0,  # Pas de GPU comme spécifié
                        "num_thread": 4  # Limiter les threads pour respecter la contrainte de mémoire
                    }
                }
            )
            
            if ollama_response.status_code != 200:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Erreur Ollama: {ollama_response.text}"
                )
            
            response_data = ollama_response.json()
            summary = response_data.get("response", "Aucun résumé généré")
            
        # 5. Retourner le résumé avec les métadonnées
        return {
            "query": query,
            "summary": summary,
            "sources": [result["url"] for result in search_results["results"][:5]]
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancer avec: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)