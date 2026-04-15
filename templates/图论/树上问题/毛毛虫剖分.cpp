
vector<int> e[N];
vector<pair<int, int>> seg[N], qu[N];
int ans[Q];
int dfn[N], dep[N], nfd[N], top[N], f[N], sz[N], hc[N], pre[N], fir[N], lst2[N], rt[N];
int
void insert()
void dfs1(int u)
{
    sz[u] = 1;
    for (int v : e[u]) if (v != f[u])
    {
        dep[v] = dep[u] + 1;
        f[v] = u;
        dfs1(v);
        sz[u] += sz[v];
        if (sz[v] > sz[hc[u]]) hc[u] = v;
    }
    if (f[u]) erase(e[u], f[u]);
}
void dfs2(int u)
{
    static int id = 0;
    //dbg(u);
    if (!dfn[u])
    {
        dfn[u] = ++id;
        nfd[id] = u;
    }
    if (top[u] == u)
    {
        vector<int> stk;
        for (int v = u;v;v = hc[v])
        {
            for (int w : e[v]) if (w != hc[v])
            {
                dfn[w] = ++id;
                nfd[id] = w;
                pre[v] = id;
                cmin(fir[v], id);
                lst2[v] = id;
            }
            stk.push_back(v);
        }
        for (int i = (int)stk.size() - 2;i >= 0;i--)
        {
            cmin(fir[stk[i]], fir[stk[i + 1]]);
            cmax(lst2[stk[i]], lst2[stk[i + 1]]);
        }
        for (int i = 1;i < stk.size();i++)
        {
            cmax(pre[stk[i]], pre[stk[i - 1]]);
        }
    }
    //dbg(u);
    top[hc[u]] = top[u];
    if (hc[u]) dfs2(hc[u]);
    for (int v : e[u]) if (v != hc[u]) dfs2(top[v] = v);
}
mt19937 rnd(245);
int main()
{
    memset(fir, 0x3f, sizeof fir);
    ios::sync_with_stdio(0); cin.tie(0);
    cout << fixed << setprecision(15);
    int n, m, q, i, j;
    cin >> n >> m >> q;
    for (i = 1;i < n;i++)
    {
        int u, v;
        //cin >> u >> v;
        u = i + 1;
        v = rnd() % i + 1;
        //v = (i + 1) / 2;
        //v = i / 2 + 1;
        //dbg(u, v);
        e[u].push_back(v);
        e[v].push_back(u);
    }
    dfs1(dep[1] = 1);
    //dbg("??");
    dfs2(top[1] = 1);
    //for (i = 1;i <= n;i++) cerr << i << ": " << dfn[i] << endl;
    for (i = 1;i <= m;i++)
    {
        int u, v;
        //cin >> u >> v;
        u = rnd() % n + 1;
        v = rnd() % n + 1;
        int uu = u, vv = v;
        //dbg(uu, vv);
        auto& w = seg[i];
        while (top[u] != top[v])
        {
            if (dep[top[u]] < dep[top[v]]) swap(u, v);
            w.push_back({fir[top[u]], pre[u]});
            //else w.push_back({fir[top[u]], lst2[top[u]]});
            if (hc[u]) w.push_back({dfn[hc[top[u]]], dfn[hc[u]]});
            else if (top[u] != u) w.push_back({dfn[hc[top[u]]], dfn[u]});
            //dbg(u, v, w);
            //[fir[top[u]],lst[u]]
            u = f[top[u]];
        }
        if (dep[u] < dep[v]) swap(u, v);
        w.push_back({fir[v], pre[u]});
        //else if (!hc[u]) w.push_back({fir[v], lst2[v]});
        //dbg(v, lst2[v], fir[v]);
        if (hc[u]) w.push_back({dfn[hc[v]], dfn[hc[u]]});
        else if (u != v) w.push_back({dfn[hc[v]], dfn[u]});
        //dbg(w);
        w.push_back({dfn[v], dfn[v]});
        if (f[v]) w.push_back({dfn[f[v]], dfn[f[v]]});
        erase_if(w, [&](const auto& x) {return x.first > x.second;});
        //int len = 0;
        //for (auto [l, r] : w) len += r - l + 1;
        //for (auto [l, r] : w)
        //{
        //    for (int j = l;j <= r;j++) cerr << nfd[j] << ' ';cerr << " | ";
        //}
        //cerr << endl;
        //int tl = 0;
        //set<int> s = {uu, vv};
        //while (uu != vv)
        //{
        //    if (dep[uu] < dep[vv]) swap(uu, vv);
        //    s.insert(all(e[uu]));s.insert(f[uu]);uu = f[uu];
        //}
        //s.insert(all(e[uu]));
        //if (f[uu]) s.insert(f[uu]);
        ////dbg(s);
        //assert(len == s.size());
    }
    for (i = 1;i <= q;i++)
    {
        int l, r;
        cin >> l >> r;
        qu[l].push_back({r, i});
    }
    for (i = m;i;i--)
    {

    }
    for (i = 1;i <= q;i++) cout << ans[i] << '\n';
    //cerr << "??\n";
}

