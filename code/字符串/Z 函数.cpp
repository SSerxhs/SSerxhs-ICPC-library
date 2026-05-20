vector<int> Z(const string &s)
{
	int n = s.size(), i, l, r;
	vector<int> z(n);
	z[0] = n;
	for (i = 1, l = r = 0; i < n; i++)
	{
		if (i <= r && z[i - l] < r - i + 1) z[i] = z[i - l];
		else
		{
			z[i] = max(0, r - i + 1);
			while (i + z[i] < n && s[i + z[i]] == s[z[i]]) ++z[i];
		}
		if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
	}
	return z;
}

