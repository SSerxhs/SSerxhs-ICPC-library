vector<int> manacher(const string &t)
{
	string S = "$#";
	int n = t.size(), i, r = 1, m = 0;
	for (i = 0; i < n; i++) S += t[i], S += '#';
	S += '#';
	char *s = S.data() + 2;
	n = n * 2 - 1;
	vector<int> ex(n);
	ex[0] = 2;
	for (i = 1; i < n; i++)
	{
		ex[i] = i < r ? min(ex[m * 2 - i], r - i + 1) : 1;
		while (s[i + ex[i]] == s[i - ex[i]]) ++ex[i];
		if (i + ex[i] - 1 > r) r = i + ex[m = i] - 1;
	}
	for (int &x : ex) --x;
	return ex;
}
pair<vector<int>, vector<pair<int, int>>> distinct_palindrome(const string &t)
// [l,r)
{
    string S = "$#";
    int n = t.size(), i, r = 1, m = 0;
    for (i = 0; i < n; i++) (S += t[i]) += '#';
    S += '#';
    char *s = S.data() + 2;
    n = n * 2 - 1;
    vector<int> ex(n);
    vector<pair<int, int>> res = {{0, 1}};
    ex[0] = 2;
    for (i = 1; i < n; i++)
    {
        if (i < r) ex[i] = min(ex[m * 2 - i], r - i + 1);
        else
        {
            ex[i] = 1;
            if (!(i & 1)) res.push_back({i / 2, i / 2 + 1});
        }
        while (s[i + ex[i]] == s[i - ex[i]])
        {
            if (s[i - ex[i]] != '#')
                res.push_back({i - ex[i] >> 1, i + ex[i] + 2 >> 1});
            ++ex[i];
        }
        if (i + ex[i] - 1 > r) r = i + ex[m = i] - 1;
    }
    for (int &x : ex) --x;
    return {ex, res};
}
