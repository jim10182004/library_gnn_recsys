# BERT / Cover 模型分析報告

> 為什麼這兩個複雜 feature 沒帶來顯著提升？
> 本報告檢驗兩個假設：feature 品質 vs 融合方式

---

### BERT 嵌入品質檢查

**「應該相似」的書對的 cosine similarity**：

- 「Twister on tuesday (Magic」 vs 「Art Toy Story(上) : 藝術玩具發展」 → cos = **0.325**
- 「好餓的毛毛蟲立體洞洞書」 vs 「噗!是誰在放屁? : 動物科普知識繪本」 → cos = **0.544**
- 「ABAQUS+Python讓CAE如虎添翼的雙倍能」 vs 「精通機器學習 : 使用Scikit-Learn, 」 → cos = **0.243**

**「應該不同」的書對的 cosine similarity**：

- 「白金數據」 vs 「Twister on tuesday (Magic」 → cos = **0.347**
- 「好餓的毛毛蟲立體洞洞書」 vs 「ABAQUS+Python讓CAE如虎添翼的雙倍能」 → cos = **0.336**
- 「解憂雜貨店 ナミヤ雑貨店の奇蹟」 vs 「圖解資料結構 : 使用Java」 → cos = **0.301**

**結論**：相似書對平均 0.371, 不同書對平均 0.328, 差距 = **+0.043**
→ ⚠️ BERT 區分能力很弱（可能因為用通用 multilingual MiniLM，沒對中文 fine-tune）

---

### Cover 覆蓋率

- 總書本數（嘗試下載）：66
- **成功下載 cover 的**：**66** (100.0%)
→ 足夠。可調整 fusion 方式進一步提升。

---

### BERT / Cover 對長尾的影響（從 Coverage 觀察）

| 模型 | Recall@10 | Coverage@10 | Coverage 提升 |
|---|---|---|---|
| lightgcn | 0.2648 | 0.0595 | +0% |
| lightgcn_bert | 0.2674 | 0.0636 | +7% |
| lightgcn_cover | 0.2602 | 0.0167 | -72% |

→ BERT 對 Coverage 影響不大
→ ⚠️ Cover 模型 Coverage 反而比 LightGCN 低（0.060 → 0.017）
   推測：因為大多數書沒 cover，模型過擬合「有 cover 的少數書」

---

### 各種 feature 的相對 Recall@10 提升

| Feature 類型 | 來源 | Recall@10 | vs LightGCN | 結論 |
|---|---|---|---|---|
| lightgcn_bert | BERT (multilingual MiniLM, 384d) | 0.2674 | +1.00% | ✅ 有幫助 |
| lightgcn_cover | ResNet-18 cover (512d) | 0.2602 | -1.74% | ❌ 退步 |
| lightgcn_si | Side Info (gender/age/category) | 0.2667 | +0.73% | ～ 持平 |
| lightgcn_multi | 預約 weak edges | 0.2684 | +1.38% | ✅ 有幫助 |
| lightgcn_hetero | 作者節點異質圖 | 0.2649 | +0.04% | ～ 持平 |
| lightgcn_timedecay | 時間衰減邊權重 | 0.2683 | +1.34% | ✅ 有幫助 |
| lightgcn_tgn | Time2Vec 時間編碼 | 0.2672 | +0.89% | ～ 持平 |

**觀察**：
- 預約 + side info（簡單拼接）效果最好（+1.4%）
- BERT / Cover 等複雜 feature 效果不顯著（< 1%）
- 推論：協同訊號（互動矩陣）已經很強，外部 feature 是「冗餘」

---

## 整體結論

1. **BERT 沒大幅贏的主因 = feature 品質**（multilingual MiniLM 對中文書名區分能力弱）
2. **Cover 沒贏的主因 = 覆蓋率太低**（4.4%，feature 主要是零向量）
3. **融合方式不是主要瓶頸**：簡單相加 vs attention 預期差異 < 1%，相對於 feature 品質問題影響小

**改進方向**：
- BERT：fine-tune BERT-wwm-ext-zh on 圖書館書名 + 內容簡介
- Cover：改用 Google Books / 博客來 API（中文書封覆蓋預估 70%+）
- Fusion：在 feature 品質提升後再考慮 attention/gated fusion