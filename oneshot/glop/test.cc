#include <ortools/linear_solver/linear_solver.h>

namespace operations_research {
void example() {
		MPSolver* solver(MPSolver::CreateSolver("GLOP"));

		MPVariable* const x = solver->MakeNumVar(0.0, 1, "x");
		MPVariable* const y = solver->MakeNumVar(0.0, 2, "y");

		LOG(INFO) << "Number of variables = " << solver->NumVariables();

		MPConstraint* const ct = solver->MakeRowConstraint(0.0, 2.0, "ct");
		ct->SetCoefficient(x, 1);
		ct->SetCoefficient(y, 1);

		LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

		MPObjective* const objective = solver->MutableObjective();
		objective->SetCoefficient(x, 3);
		objective->SetCoefficient(y, 1);
		objective->SetMaximization();

		solver->Solve();

		LOG(INFO) << "Solution: " << std::endl;
		LOG(INFO) << "Objective vale = " << objective->Value();
		LOG(INFO) << "x = " << x->solution_value();
		LOG(INFO) << "y = " << y->solution_value();
}
}

int main() {
		operations_research::example();
		return EXIT_SUCCESS;
}
