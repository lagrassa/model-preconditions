task_names = ["RodInBox", "RodInDrawer"]
anchor_baseline = {'label': "ours : one queue",
    'RodInBox': {'plan_found': [1] * 10,
                                   'plan_success': [1,1,1,1,1,1,1,1,0] * 10},
                   'RodInDrawer': {'plan_found': [1] * 10,
                                'plan_success': [0,1,1,1,1,1,1,0,1,1] * 10}}

random_baseline = {'label': "random model",
    'RodInBox': {'plan_found': [1] * 10,
                                   'plan_success': [1,0,1,0,1,1,1,0,0,0]},
                   'RodInDrawer': {'plan_found': [1,0,1,1,1,0,0,0,1,0],
                                'plan_success': [1,1,1,0,0,0]}}

ours = {"label": "ours : multiple queues",
    'RodInBox': {'plan_found': [1] * 10,
                                'plan_success': [1] * 10},
                   'RodInDrawer': {'plan_found': [1] * 10,
                                   'plan_success': [1,1,1,1,0,1,1,1,1] * 10}}

simulator_only = {"label": "simulator",
        'RodInBox': {'plan_found': [1] * 10,
                     'plan_success': [0,1,1,0,0,1,0,1,0,1] * 10},
        'RodInDrawer': {'plan_found': [1,0,1,0,1,1,1,0,0,1],
                        'plan_success': [1,0,0,1,0,1] * 10}}

analytical_only_rod_and_robot = {"label": "analytical (rod & robot)",
                  'RodInBox': {'plan_found': [1] * 10,
                               'plan_success': [0,1,0,1,0,0,0,1,0,1]},
                  'RodInDrawer': {'plan_found': [eps] * 10,
                                  'plan_success': [eps] * 10}}

analytical_only_rod_and_drawer = {"label": "analytical (drawer & robot)",
                                 'RodInBox': {'plan_found': [eps] * 10,
                                              'plan_success': [eps] * 10},
                                 'RodInDrawer': {'plan_found': [eps] * 10,
                                                 'plan_success': [eps] * 10}}

