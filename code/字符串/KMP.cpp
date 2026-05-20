template<class T> struct str//[0,n)
{
    int n;
    vector<int> nxt;//长度 - 1，就是 father
    T s;
    str(const T &s) : n(s.size()), nxt(n, -1), s(s)
    {
        int i, j = -1;
        for (i = 1; i < n; i++)
        {
            while (j != -1 && s[i] != s[j + 1]) j = nxt[j];
            nxt[i] = j += s[i] == s[j + 1];
        }
    }
    vector<int> match(const T &t)//find s(str) in t (start pos)
    {
        int m = t.size(), i, j = -1;
        vector<int> r;
        for (i = 0; i < m; i++)
        {
            while (j != -1 && t[i] != s[j + 1]) j = nxt[j];
            if ((j += t[i] == s[j + 1]) == n - 1) j = nxt[j], r.push_back(i - n + 1);
        }
        return r;
    }
};

	
