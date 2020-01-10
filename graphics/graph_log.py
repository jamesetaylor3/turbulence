import liveplot

losses = [int(line[:1]) for line in open("../../losses.txt", "r").readlines()]
fuel = [int(line[:1]) for line in open("../../fuel.txt", "r").readlines()]
turb = [int(line[:1]) for line in open("../../turb.txt", "r").readlines()]
angle = [int(line[:1]) for line in open("../../angle.txt", "r").readlines()]

liveplot.p_losses(losses)
liveplot.p_fuel_costs(fuel)
liveplot.p_turb_costs(turb)
liveplot.p_time_diff_costs(angle)

raw_input("Press enter to quit program")
