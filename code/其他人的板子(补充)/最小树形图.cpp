struct RollbackUnionFind
{
    vector<pair<int, int>> st;
    vector<int> f;
    RollbackUnionFind(int n) : f(n, -1) { }
    int find(int u) { return f[u] < 0 ? u : find(f[u]); }
    bool merge(int u, int v)
    {
        if ((u = find(u)) == (v = find(v))) return false;
        if (f[u] < f[v]) swap(u, v);
        st.emplace_back(u, f[u]);
        f[v] += f[u];
        f[u] = v;
        return true;
    }
    void rollback(int t)
    {
        while (st.size() > t)
        {
            auto [u, v] = st.back();
            st.pop_back();
            f[f[u]] -= v;
            f[u] = v;
        }
    }
};
struct Skew
{
    int u, v;
    ll w, lazy;
    Skew *lc, *rc;
    static Skew *merge(Skew *x, Skew *y)
    {
        if (!x) return y;
        if (!y) return x;
        if (x->w > y->w) swap(x, y);
        x->push();
        x->rc = merge(x->rc, y);
        swap(x->lc, x->rc);
        return x;
    }
    Skew(tuple<int, int, ll> e) : lazy(0)
    {
        tie(u, v, w) = e;
        lc = rc = nullptr;
    }
    void add(ll x)
    {
        w += x;
        lazy += x;
    }
    void push()
    {
        if (lc) lc->add(lazy);
        if (rc) rc->add(lazy);
        lazy = 0;
    }
    Skew *pop()
    {
        push();
        return merge(lc, rc);
    }
};
pair<ll, vector<int>> directed_minimum_spanning_tree(int n, const vector<tuple<int, int, ll>> &edges, int s) //[0,n)
{
    ll ans = 0;
    vector<Skew *> h(n), in(n);
    RollbackUnionFind f(n), r(n);
    vector<pair<Skew *, int>> c;
    for (auto [u, v, w] : edges) h[v] = Skew::merge(h[v], new Skew({u, v, w}));
    for (int i = 0; i < n; i += 1)
    {
        if (i == s) continue;
        for (int u = i;;)
        {
            while (h[u] && r.find(h[u]->u) == r.find(u)) h[u] = h[u]->pop(); // fix by codex
            if (!h[u]) return { };
            ans += (in[u] = h[u])->w;
            in[u]->add(-in[u]->w);
            int v = r.find(in[u]->u);
            if (f.merge(u, v)) break;
            int t = r.st.size();
            while (r.merge(u, v)) {
                h[r.find(u)] = Skew::merge(h[u], h[v]);
                u = r.find(u);
                v = r.find(in[v]->u);
            }
            c.emplace_back(in[u], t);
            while (h[u] && r.find(h[u]->u) == r.find(u)) h[u] = h[u]->pop();
        }
    }
    reverse(all(c));
    for (auto [p, t] : c)
    {
        int u = r.find(p->v);
        r.rollback(t);
        int v = r.find(in[u]->v);
        in[v] = exchange(in[u], p);
    }
    vector<int> res(n, -1);
    for (int i = 0; i < n; i += 1) res[i] = i == s ? i : in[i]->u;
    return {ans, res};
}
