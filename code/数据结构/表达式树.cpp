struct Q
{
	char ch;
	int prec;
	bool right;
};
tuple<vector<array<int, 2>>, vector<char>, int> parse_expr(string s, vector<Q> op) {
	static int idx[128];
	int maxp = 0, pos = 0, n, err = 0, i;
	{
		string t;
		for (char c : s)
		{
			if (t.size() && isdigit(t.back()) && isdigit(c)) t += '#';
			t += c;
		}
		swap(s, t);
		n = s.size();
	}
	for (i = 0; i < op.size(); ++i)
	{
		idx[op[i].ch] = i + 1;
		cmax(maxp, op[i].prec);
	}
	op.push_back({'#', ++maxp, 0});
	idx['#'] = op.size();
	vector<array<int, 2>> c(1);
	vector<char> ch(1);
	auto node = [&](char x) {
		c.push_back({0, 0});
		ch.push_back(x);
		return c.size() - 1;
	};
	function<int(int)> parse = [&](int lv) -> int {
		int u;
		if (lv > maxp)
		{
			if (pos < n && s[pos] == '(')
			{
				pos++;
				u = parse(1);
				if (err |= (pos >= n || s[pos++] != ')')) return 0;
				return u;
			}
			else if (pos < n && isdigit(s[pos])) return u = node(s[pos++]);
			else return err = 1, 0;
		}
		else
		{
			u = parse(lv + 1);
			while (!err && pos < n)
			{
				char ch = s[pos];
				int i = idx[ch] - 1;
				if (i >= 0 && op[i].prec == lv)
				{
					++pos;
					int v = node(ch), w = parse(lv + !op[i].right);
					c[v] = {u, w};
					u = v;
				}
				else break;
			}
			return u;
		}
	};
	int root = parse(0);
	for (auto [ch, _, __] : op) idx[ch] = 0;
	if (err || pos != n) return {{ }, { }, 0};
	return {c, ch, root};
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	string s;
	getline(cin, s);
	vector<Q> op = {
		{'|', 1, 0},
		{'&', 2, 0},
	};
	auto [c, ch, root] = parse_expr(s, op);
	assert(root);
	function<array<int, 3>(int)> dfs = [&](int u)->array<int, 3> {
		if (isdigit(ch[u])) return {ch[u] - '0', 0, 0};
		auto [l, r1, r2] = dfs(c[u][0]);
		if (ch[u] == '|')
		{
			if (l) return {1, r1, r2 + 1};
			auto [r, r3, r4] = dfs(c[u][1]);
			return {r, r1 + r3, r2 + r4};
		}
		else
		{
			if (!l) return {0, r1 + 1, r2};
			auto [r, r3, r4] = dfs(c[u][1]);
			return {r, r1 + r3, r2 + r4};
		}
	};
	auto [r0, r1, r2] = dfs(root);
	cout << r0 << endl << r1 << ' ' << r2 << endl;
}

