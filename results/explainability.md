# 推薦可解釋性分析

**目的**：對 LightGCN 的推薦結果，分解出每個輸入種子書對最終推薦分數的貢獻，
證明推薦不是黑箱，而是可數學分解的。

**方法**：
- 合成讀者向量 $u = \frac{1}{N} \sum_{j=1}^{N} \hat{e}_j$（$\hat{e}_j$ 為 seed $j$ 的 normalised embedding）
- 對推薦書 $i$ 的 cosine similarity 分數 $s_i = \hat{u} \cdot \hat{e}_i$
- $s_i$ 可線性分解為 $\sum_j \frac{1}{N} \hat{e}_j \cdot \hat{e}_i$（每個 seed 的貢獻）

## Case：日系推理小說迷

**Seed books**:
- (1) 白金數據
- (2) 解憂雜貨店 ナミヤ雑貨店の奇蹟

**Top-5 recommendations 與每個 seed 的貢獻**：

| Rank | 推薦書 | Final Score | Seed1 貢獻 | Seed2 貢獻 |
|---|---|---|---|---|
| 1 | 醫學系在幹嘛? : 笑中帶淚的超狂醫界人生 | 0.8074 | +0.2897 | +0.3266 |
| 2 | 名偵探的枷鎖 | 0.8006 | +0.4571 | +0.1540 |
| 3 | 祕史 | 0.7975 | +0.2513 | +0.3574 |
| 4 | 去唱卡拉OK吧! カラオケ行こ! | 0.7904 | +0.1966 | +0.4067 |
| 5 | 跟著奈良美智去旅行 | 0.7890 | +0.2161 | +0.3862 |

**主導 seed 解讀**：

- 推薦 #1 **醫學系在幹嘛? : 笑中帶淚的超狂醫界人生** 主要來自 Seed 2「解憂雜貨店 ナミヤ雑貨店の奇蹟」（佔 53.0%）
- 推薦 #2 **名偵探的枷鎖** 主要來自 Seed 1「白金數據」（佔 74.8%）
- 推薦 #3 **祕史** 主要來自 Seed 2「解憂雜貨店 ナミヤ雑貨店の奇蹟」（佔 58.7%）
- 推薦 #4 **去唱卡拉OK吧! カラオケ行こ!** 主要來自 Seed 2「解憂雜貨店 ナミヤ雑貨店の奇蹟」（佔 67.4%）
- 推薦 #5 **跟著奈良美智去旅行** 主要來自 Seed 2「解憂雜貨店 ナミヤ雑貨店の奇蹟」（佔 64.1%）

## Case：兒童英文書

**Seed books**:
- (1) Twister on tuesday (Magic Tree House#23)
- (2) Toy story read-along storybook and cd co
- (3) Follow me, Mittens

**Top-5 recommendations 與每個 seed 的貢獻**：

| Rank | 推薦書 | Final Score | Seed1 貢獻 | Seed2 貢獻 | Seed3 貢獻 |
|---|---|---|---|---|---|
| 1 | Crying in H Mart : a memoir | 0.9547 | +0.2449 | +0.3238 | +0.2817 |
| 2 | What is the story of Doctor Who? | 0.9515 | +0.2567 | +0.3189 | +0.2720 |
| 3 | Dragons and mythical creatures | 0.9476 | +0.2617 | +0.3110 | +0.2715 |
| 4 | Stripes and spots | 0.9434 | +0.2504 | +0.3186 | +0.2715 |
| 5 | Big dog... little dog | 0.9400 | +0.2781 | +0.2998 | +0.2596 |

**主導 seed 解讀**：

- 推薦 #1 **Crying in H Mart : a memoir** 主要來自 Seed 2「Toy story read-along storybook and cd co」（佔 38.1%）
- 推薦 #2 **What is the story of Doctor Who?** 主要來自 Seed 2「Toy story read-along storybook and cd co」（佔 37.6%）
- 推薦 #3 **Dragons and mythical creatures** 主要來自 Seed 2「Toy story read-along storybook and cd co」（佔 36.8%）
- 推薦 #4 **Stripes and spots** 主要來自 Seed 2「Toy story read-along storybook and cd co」（佔 37.9%）
- 推薦 #5 **Big dog... little dog** 主要來自 Seed 2「Toy story read-along storybook and cd co」（佔 35.8%）

## Case：職場成長

**Seed books**:
- (1) 原子習慣 : 細微改變帶來巨大成就的實證法則
- (2) 拖延心理學 : 為什麼我老是愛拖延?是與生俱來的壞習慣, 還是身不由己?
- (3) 目標 : 簡單有效的常識管理

**Top-5 recommendations 與每個 seed 的貢獻**：

| Rank | 推薦書 | Final Score | Seed1 貢獻 | Seed2 貢獻 | Seed3 貢獻 |
|---|---|---|---|---|---|
| 1 | 推销员之死 | 0.9104 | +0.2757 | +0.1250 | +0.3157 |
| 2 | 元宇宙革命与矩阵陷阱 : 科技大集成和文明大考 | 0.9093 | +0.2790 | +0.1203 | +0.3162 |
| 3 | 倘若有柚子 | 0.9075 | +0.2792 | +0.1213 | +0.3136 |
| 4 | 紫禁城六百年 : 帝王之轴 | 0.9029 | +0.2808 | +0.1092 | +0.3205 |
| 5 | 绘画可以速成? : 技巧之外一定要看的绘画思维法 | 0.9005 | +0.2900 | +0.1019 | +0.3167 |

**主導 seed 解讀**：

- 推薦 #1 **推销员之死** 主要來自 Seed 3「目標 : 簡單有效的常識管理」（佔 44.1%）
- 推薦 #2 **元宇宙革命与矩阵陷阱 : 科技大集成和文明大考** 主要來自 Seed 3「目標 : 簡單有效的常識管理」（佔 44.2%）
- 推薦 #3 **倘若有柚子** 主要來自 Seed 3「目標 : 簡單有效的常識管理」（佔 43.9%）
- 推薦 #4 **紫禁城六百年 : 帝王之轴** 主要來自 Seed 3「目標 : 簡單有效的常識管理」（佔 45.1%）
- 推薦 #5 **绘画可以速成? : 技巧之外一定要看的绘画思维法** 主要來自 Seed 3「目標 : 簡單有效的常識管理」（佔 44.7%）
