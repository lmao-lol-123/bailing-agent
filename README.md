# Engineering RAG Assistant

[涓枃](./README.md) | [English](./README.en.md)

闈㈠悜宸ョ▼鏂囨。鐨勮交閲忕骇 RAG 鍔╂墜锛屽綋鍓嶉樁娈典互鍚庣鑳藉姏涓烘牳蹇冿紝寮鸿皟鏈湴鍙繍琛屻€佹绱㈠彲瑙ｉ噴銆佸洖绛斿甫寮曠敤銆佹祦寮忚緭鍑虹ǔ瀹氥€?
## 鍔熻兘鐗规€?
- 澶氭牸寮忔帴鍏ワ細鏀寔 Web銆丳DF銆乄ord `.docx`銆丮arkdown銆丆SV銆丣SON銆乀XT
- 鍚庣浼樺厛锛氬洿缁?ingestion銆乧hunking銆乺etrieval銆乬eneration銆乪valuation 鍜?API 绋冲畾鎬у缓璁?- 娣峰悎妫€绱細FAISS 鍚戦噺鍙洖缁撳悎 BM25 璇嶆硶鍙洖銆佹煡璇㈡敼鍐欍€佽矾鐢卞拰 rerank
- 寮曠敤鍙拷婧細鍥炵瓟闄勫甫 citation锛屾绱?metadata 璐┛绱㈠紩銆佸彫鍥炲拰鐢熸垚
- 娴佸紡闂瓟锛歚POST /ask/stream` 鍩轰簬 SSE 杈撳嚭 token銆乻ources 鍜屽畬鎴愪簨浠?- 浼氳瘽鎸佷箙鍖栵細SQLite 瀛樺偍浼氳瘽涓庢秷鎭紝鏀寔澶氳疆杩介棶

## 鎶€鏈爤

| 缁勪欢 | 鎶€鏈?|
| --- | --- |
| API | FastAPI, Pydantic, StreamingResponse |
| Ingestion | LangChain loaders, pymupdf4llm, Unstructured |
| Chunking | 缁撴瀯浼樺厛鍒囧潡 + 棰勭畻绾︽潫鍒囧垎 |
| Retrieval | sentence-transformers, FAISS, rank-bm25 |
| Generation | DeepSeek API via OpenAI-compatible SDK |
| Persistence | SQLite, local JSON snapshots |
| Testing | pytest, pytest-asyncio, FastAPI TestClient |

## 褰撳墠鍚庣鑳藉姏

- 鏂囨。瀵煎叆鍚庝細鍋氬綊涓€鍖栵紝骞跺皢蹇収鍐欏叆 `data/processed/`
- 妫€绱㈤摼璺凡鏀寔鏌ヨ璺敱銆佺‘瀹氭€ф煡璇㈡墿灞曘€佹贩鍚堝彫鍥炪€佸惎鍙戝紡 rerank 鍜屽畾鍚戦噸璇?- 瀛愬潡鍛戒腑鍚庝細灏介噺鍥炲～鐖跺潡姝ｆ枃锛屾彁鍗囧洖绛斾笂涓嬫枃瀹屾暣鎬?- 褰撹瘉鎹笉瓒虫椂锛岀敓鎴愬眰浼氭槑纭繑鍥炴棤娉曠‘璁わ紝鑰屼笉鏄紪閫犵瓟妗?- DeepSeek 鐪熷疄 API key 鑱旇皟宸查獙璇侀€氳繃锛屾祦寮忕敓鎴愰摼璺彲鐢?
## 蹇€熷紑濮?
鎺ㄨ崘浣跨敤浠撳簱鏈湴铏氭嫙鐜锛?
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

澶嶅埗鐜鍙橀噺妯℃澘骞跺～鍐?DeepSeek 閰嶇疆锛?
```powershell
Copy-Item .env.example .env
```

`.env` 鍏抽敭椤癸細

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

鍚姩 API锛?
```powershell
.venv\Scripts\activate
uvicorn backend.src.api.main:app --reload
```

甯哥敤鍏ュ彛锛?
- Web/API 鍏ュ彛锛歚http://127.0.0.1:8000/`
- OpenAPI 鏂囨。锛歚http://127.0.0.1:8000/docs`

## 甯哥敤鍛戒护

鏋勫缓绱㈠紩锛?
```powershell
.venv\Scripts\activate
python -m backend.scripts.build_index data\sample.txt
```

CLI 鎻愰棶锛?
```powershell
.venv\Scripts\activate
python -m backend.scripts.ask "What does this document say?"
```

杩愯娴嬭瘯锛?
```powershell
.venv\Scripts\activate
python -m pytest -q
```

## API 姒傝

- `GET /health`
- `POST /ingest/files`
- `POST /ingest/url`
- `GET /documents`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /ask/stream`

`POST /ask/stream` 鏀寔 `question`銆佸彲閫?`top_k`銆佸彲閫?`session_id` 鍜屽彲閫?`metadata_filter`銆?
## 椤圭洰缁撴瀯

```text
bailing-agent/
|- backend/
|  |- scripts/
|  |- src/
|  |  |- api/
|  |  |- core/
|  |  |- eval/
|  |  |- generate/
|  |  |- ingest/
|  |  |- models/
|  |  `- retrieve/
|  `- tests/
|- data/
|- docs/
|- frontend/
|- storage/
|- .github/workflows/ci.yml
|- CHANGELOG.md
|- README.en.md
`- README.md
```

## 楠岃瘉鐘舵€?
- 鍚庣瀹屾暣娴嬭瘯锛歚60 passed`
- DeepSeek 鐪熷疄娴佸紡鐢熸垚鑱旇皟锛氬凡閫氳繃
- CI锛欸itHub Actions 鍦?`master`銆乣main` 鍜?`codex/**` 鍒嗘敮瑙﹀彂娴嬭瘯

## 浠撳簱绾﹀畾

- `data/uploads/`銆乣data/processed/` 鍜?`storage/` 灞炰簬鏈湴杩愯浜х墿锛屼笉鎻愪氦鍒?GitHub
- 褰撳墠闃舵鍓嶇涓虹淮鎶ゆā寮忥紝鏈粨搴撹凯浠ｄ互 backend 涓轰富
- 濡傛灉璇佹嵁涓嶈冻锛岀郴缁熶紭鍏堣繑鍥炰笉纭畾锛岃€屼笉鏄嚜鐢卞彂鎸?
## 鍙傝€?
- 椤圭洰瑙勫垯锛歔AGENTS.md](./AGENTS.md)
- Deeptoai RAG docs: [https://rag.deeptoai.com/docs](https://rag.deeptoai.com/docs)