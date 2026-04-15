[&](const int &x) {
	if (x >= s) return true;
	s -= x;
	return false; }
