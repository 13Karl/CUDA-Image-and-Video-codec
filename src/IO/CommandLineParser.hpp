#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#ifndef COMMANDLINEPARSER_HPP
#define COMMANDLINEPARSER_HPP

class InputParser {
public:
	InputParser(int &argc, char **argv);
	//author iain
	const std::string& getCmdOption(const std::string &option) const;
	//author iain
	bool cmdOptionExists(const std::string &option) const;

	void printHelp();
private:
	std::vector <std::string> _tokens;
};

#endif