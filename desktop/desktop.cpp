#include <iostream>
#include <fstream>
using namespace std;

int main() {
	ifstream ifs;
	ifs.open("./path.in");
	if (!ifs.is_open())
	{
		cout << "path.in is missing!" << endl;
		ifs.close();
		return 1;
	}

	string sLine;
	getline(ifs, sLine);
	if (sLine.empty())
	{
		cout << "no desktop folder!" << endl;
	}

	string cmd("start \"\" \"");
	cmd.append(sLine).append("\"");
	cout << cmd << endl;
	system(cmd.c_str());
	ifs.close();
	return 0;
}