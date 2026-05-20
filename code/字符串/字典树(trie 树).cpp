struct trie
{
    const static int N = 1e6 + 2, M = 52;
    int c[N][M], sz[N];//sz 维护有多少个以当前字符串为前缀的字符串。
    int cnt = 1;
    void insert(string s)
    {
        int u = 1;
        ++sz[u];
        for (char ch : s)
        {
            assert(ch >= 0 && ch < M);
            int &v = c[u][ch];
            if (!v) v = ++cnt;
            u = v;
            ++sz[u];
        }
        //此时 u 是字符串结束位置。你可以在此存储结点信息。
    }
    int match(string s)//返回字符串结束位置。可能为 0。
    {
        int u = 1;
        for (char ch : s)
        {
            assert(ch >= 0 && ch < M);
            u = c[u][ch];
            if (!u) return 0;
        }
        return u;
    }
    void clear()
    {
        memset(c, 0, (cnt + 1) * sizeof c[0]);
        memset(sz, 0, (cnt + 1) * sizeof sz[0]);
        cnt = 1;
    }
} t;
