using JuMP
import MathOptInterface as MOI
import Madeline

# Create a JuMP model with Madeline
model = Model(Madeline.Optimizer)

# Get the backend
backend = JuMP.backend(model)
println("Backend type: ", typeof(backend))

# Try to access the inner optimizer
println("Optimizer type: ", typeof(backend.optimizer))

# In JuMP, RawOptimizerAttribute gets forwarded to the backend
println("\n--- Using JuMP set_attribute ---")
set_attribute(model, "scaling", false)
println("scaling (via JuMP get_attribute): ", get_attribute(model, "scaling"))

# Try setting via MOI directly on the backend
println("\n--- Using MOI.set on backend ---")
MOI.set(backend, MOI.RawOptimizerAttribute("scaling"), false)
println("scaling (via MOI.get): ", MOI.get(backend, MOI.RawOptimizerAttribute("scaling")))

# Check the optimizer's internal state
# The backend.optimizer is a LazyBridgeOptimizer wrapping a CachingOptimizer wrapping Madeline.Optimizer
inner = backend.optimizer.model.optimizer
println("\n--- Innermost optimizer ---")
println("Type: ", typeof(inner))
println("Settings: ", inner.settings)
